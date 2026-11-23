// RTMDet + RTMPose top-down pose estimation ROS2 node (C++).
//
// C++ migration of person_pose_python/pose_node.py. Inference is done through the
// MMDeploy C++ SDK (Detector = RTMDet, PoseDetector = RTMPose), which compiles
// against TensorRT on Jetson / ONNXRuntime-CPU on a dev box. Everything else
// (camera capture, MJPEG preview, JSON publishing) is a direct port.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <person_pose_interfaces/msg/person_pose_array.hpp>

#include <mmdeploy/detector.hpp>
#include <mmdeploy/pose_detector.hpp>

using namespace std::chrono_literals;

namespace {

// COCO-17 skeleton edges, used only for the preview overlay.
const std::vector<std::pair<int, int>> kCocoSkeleton = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
    {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {1, 2},  {0, 1},
    {0, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6}};

std::string expand_user(const std::string& path) {
  if (!path.empty() && path[0] == '~') {
    const char* home = std::getenv("HOME");
    if (home) return std::string(home) + path.substr(1);
  }
  return path;
}

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool file_exists(const std::string& path) {
  return std::ifstream(path).good();
}

}  // namespace

class PoseEstimationNode : public rclcpp::Node {
 public:
  // Naming-contract migration (perception domain): fully-qualified node
  // name /perception/person_pose - public endpoints below use absolute
  // paths (leading /) rather than relying on this namespace, matching the
  // proposal's own "public endpoints do not depend on a particular node's
  // name" rule.
  PoseEstimationNode() : Node("person_pose", "perception") {
    declare_parameter<int>("camera_index", 0);
    declare_parameter<bool>("use_fallback", false);
    declare_parameter<std::string>("fallback_image_path", "");
    // 10 was faster than anything downstream actually consumes -
    // person_body ticks at 2Hz, person_identity at 5Hz,
    // person_face at 1Hz - so this was paying full RTMDet+RTMPose
    // cost (the single most expensive component in the whole perception
    // stack, confirmed live at 549% CPU) roughly 2x more often than
    // anything needed. 5Hz matches the fastest actual consumer
    // (person_identity) with no downstream staleness. Still a runtime-
    // tunable param (-p target_fps:=X) if a future consumer needs faster.
    declare_parameter<double>("target_fps", 5.0);
    // MMDeploy SDK parameters
    declare_parameter<std::string>("det_model_path", "");
    declare_parameter<std::string>("pose_model_path", "");
    declare_parameter<std::string>("device", "cpu");  // "cpu" or "cuda"
    // 0.3 was letting real non-human false positives through as "person"
    // detections (confirmed live, downstream in person_body) - 0.5
    // is a more conservative floor for what counts as a confident person
    // box. Still a runtime-tunable param (-p det_score_thr:=X), this only
    // changes the default.
    declare_parameter<double>("det_score_thr", 0.5);
    // COCO-17 has no hand/finger keypoints at all - a box RTMDet mistook a
    // hand for "person" should come back with almost none of these 17
    // scoring confidently (nothing in it maps to a shoulder/hip/elbow/etc),
    // where a real (even partly occluded) body reliably clears several.
    // Reject anything under this count entirely - not published, not drawn
    // - rather than just flagging it, since a bare bbox with no real
    // skeleton isn't useful to any downstream consumer (person_body,
    // person_face) either.
    declare_parameter<int>("min_keypoints", 5);
    declare_parameter<double>("keypoint_score_thr", 0.3);
    declare_parameter<std::string>("frame_id", "camera");
    // Input source: "camera" = direct USB capture (OpenCV/V4L2),
    //               "topic"  = subscribe to a sensor_msgs/Image topic (e.g. D455).
    declare_parameter<std::string>("input_source", "camera");
    declare_parameter<std::string>("image_topic", "/camera/color/image_raw");

    const int camera_index = get_parameter("camera_index").as_int();
    const bool use_fallback = get_parameter("use_fallback").as_bool();
    input_source_ = get_parameter("input_source").as_string();
    const std::string image_topic = get_parameter("image_topic").as_string();
    const std::string fallback_path = get_parameter("fallback_image_path").as_string();
    const double target_fps = get_parameter("target_fps").as_double();
    const std::string det_model = get_parameter("det_model_path").as_string();
    const std::string pose_model = get_parameter("pose_model_path").as_string();
    const std::string device = get_parameter("device").as_string();
    det_score_thr_ = static_cast<float>(get_parameter("det_score_thr").as_double());
    min_keypoints_ = get_parameter("min_keypoints").as_int();
    kp_score_thr_ = static_cast<float>(get_parameter("keypoint_score_thr").as_double());
    frame_id_ = get_parameter("frame_id").as_string();

    // Naming-contract migration: /pose/keypoints -> /perception/people/poses.
    publisher_ = create_publisher<person_pose_interfaces::msg::PersonPoseArray>(
        "/perception/people/poses", 10);
    // Bounding boxes + skeleton overlay, same convention as every other
    // annotated-view topic in this workspace (/face/annotated,
    // /person_reid/annotated, /identity/annotated) - viewable via the
    // dashboard's Graph tab or Foxglove instead of the old standalone MJPEG
    // HTTP preview this replaces. BEST_EFFORT (sensor-data QoS), not the
    // default RELIABLE - same fix as those other annotated topics: a large,
    // frequent frame stalls entirely for any RELIABLE-requesting reader on
    // a RELIABLE publisher, confirmed live more than once this session.
    // Naming-contract migration: /pose/annotated -> /perception/debug/person_pose/image.
    annot_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "/perception/debug/person_pose/image", rclcpp::SensorDataQoS());

    if (use_fallback) {
      load_fallback(fallback_path);
    } else if (input_source_ == "topic") {
      // D455 (or any sensor_msgs/Image publisher). SensorDataQoS = best-effort,
      // which matches RealSense/most camera drivers.
      image_sub_ = create_subscription<sensor_msgs::msg::Image>(
          image_topic, rclcpp::SensorDataQoS(),
          std::bind(&PoseEstimationNode::image_cb, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "Input: subscribing to image topic '%s'", image_topic.c_str());
      load_fallback(fallback_path);  // used until the first frame arrives
    } else {
      init_camera(camera_index);
      if (!cap_.isOpened()) {
        RCLCPP_WARN(get_logger(), "Camera unavailable, falling back to static image");
        load_fallback(fallback_path);
      }
    }

    if (det_model.empty() || pose_model.empty()) {
      RCLCPP_FATAL(get_logger(),
                   "det_model_path and pose_model_path are required. See the package "
                   "README for the MMDeploy model-export steps.");
      throw std::runtime_error("missing model paths");
    }

    RCLCPP_INFO(get_logger(), "Loading MMDeploy models (device=%s)...", device.c_str());
    mmdeploy::Device mm_device{device, 0};
    detector_ = std::make_unique<mmdeploy::Detector>(mmdeploy::Model{det_model}, mm_device);
    pose_detector_ =
        std::make_unique<mmdeploy::PoseDetector>(mmdeploy::Model{pose_model}, mm_device);
    RCLCPP_INFO(get_logger(), "MMDeploy detector + pose detector ready");

    const auto period = std::chrono::duration<double>(1.0 / target_fps);
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&PoseEstimationNode::timer_cb, this));
  }

  ~PoseEstimationNode() override {
    if (cap_.isOpened()) cap_.release();
  }

 private:
  // ------------------------------------------------------------------
  // Camera helpers
  // ------------------------------------------------------------------

  void init_camera(int index) {
    cap_.open(index, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
      RCLCPP_WARN(get_logger(), "VideoCapture(%d) failed to open", index);
      cap_.release();
      return;
    }
    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    // Probe grab() — on WSL2/usbipd UVC, isOpened() is true but grab() fails.
    if (!cap_.grab()) {
      RCLCPP_WARN(get_logger(),
                  "Camera opened but grab() failed — likely WSL2/usbipd UVC issue. "
                  "Run with --ros-args -p use_fallback:=true to use a static image.");
      cap_.release();
      return;
    }
    RCLCPP_INFO(get_logger(), "Camera ready at /dev/video%d", index);
  }

  void load_fallback(const std::string& path) {
    std::vector<std::string> candidates;
    if (!path.empty()) candidates.push_back(path);
    candidates.push_back(expand_user("~/test_pose.jpg"));
    candidates.push_back(expand_user("~/test_frame.jpg"));
    candidates.push_back(expand_user("~/test.mp4"));
    candidates.push_back(expand_user("~/test_fixed.mp4"));

    for (const auto& candidate : candidates) {
      if (!file_exists(candidate)) continue;
      if (ends_with(candidate, ".mp4") || ends_with(candidate, ".avi") ||
          ends_with(candidate, ".mkv")) {
        cv::VideoCapture vc(candidate);
        cv::Mat frame;
        bool ret = vc.read(frame);
        vc.release();
        if (ret && !frame.empty()) {
          fallback_frame_ = frame;
          RCLCPP_INFO(get_logger(), "Fallback: first frame from %s", candidate.c_str());
          return;
        }
      } else {
        cv::Mat frame = cv::imread(candidate);
        if (!frame.empty()) {
          fallback_frame_ = frame;
          RCLCPP_INFO(get_logger(), "Fallback: image from %s", candidate.c_str());
          return;
        }
      }
    }
    RCLCPP_ERROR(get_logger(),
                 "No fallback image found. Provide one with "
                 "--ros-args -p fallback_image_path:=/path/to/image.jpg");
  }

  // Image-topic input (e.g. D455): convert to BGR, stash the latest frame,
  // and republish /pose/annotated directly off THIS frame using whichever
  // detections timer_cb last computed (last_people_msg_). In "topic" mode
  // frames can arrive independently of (and faster than) timer_cb's own
  // target_fps - without this, /pose/annotated would only be as fresh as
  // timer_cb's own cadence and would freeze entirely if a timer_cb pass
  // ever stalled, even though raw frames kept arriving. Same fix as
  // /face/annotated and /identity/annotated got this session, standardized
  // across every annotated topic in the workspace. Doesn't apply to
  // "camera" input mode (this node's own direct V4L2 capture) - there,
  // frames are pulled synchronously inside timer_cb itself, so there's no
  // separate raw-frame stream to decouple from in the first place.
  void image_cb(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    cv::Mat bgr;
    try {
      bgr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "cv_bridge: %s", e.what());
      return;
    }
    person_pose_interfaces::msg::PersonPoseArray people_copy;
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      topic_frame_ = bgr;
      // The whole reason this exists (see get_frame()'s own comment): every
      // downstream identity/fusion consumer needs the SAME capture-time
      // stamp this image actually carries, not whenever timer_cb() happens
      // to publish - copied here, at the earliest point this node ever
      // sees the frame, not reconstructed or re-stamped later.
      topic_frame_stamp_ = msg->header.stamp;
      people_copy = last_people_msg_;
    }
    cv::Mat vis = bgr.clone();
    draw_annotations_from_msg(vis, people_copy);
    auto annot_msg = cv_bridge::CvImage(msg->header, "bgr8", vis).toImageMsg();
    annot_pub_->publish(*annot_msg);
  }

  // Returns the newest frame from the active source, and the stamp that
  // frame's own detections/tracks must be published with - falls back to
  // the static image. Uses grab()+retrieve() (not read()) for the camera
  // to avoid the WSL2 hang.
  //
  // stamp is NEVER this node's own now() taken at some later point (e.g.
  // after detection+pose inference in timer_cb, or a whole timer period
  // after the frame actually arrived) - every consumer downstream of
  // PersonPoseArray (person_body, and person_identity two
  // hops further, see person_identity.py's own module docstring) needs the
  // ACTUAL capture time to align against a DIFFERENT topic's own per-frame
  // data by real elapsed time, not by whenever each node in the chain
  // happened to finish its own processing. "topic" mode has a real
  // upstream stamp to propagate (topic_frame_stamp_, captured the instant
  // image_cb saw it - see its own comment); "camera"/fallback modes have
  // no upstream ROS header to borrow at all, so the best available
  // approximation is stamping AT GRAB TIME, immediately after grab()
  // succeeds - still far closer to the real capture instant than stamping
  // after this whole tick's inference work runs.
  bool get_frame(cv::Mat& out, builtin_interfaces::msg::Time& stamp) {
    if (input_source_ == "topic") {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (!topic_frame_.empty()) {
        out = topic_frame_;
        stamp = topic_frame_stamp_;
        return true;
      }
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "No image received on topic yet; using fallback");
    } else if (cap_.isOpened()) {
      if (cap_.grab()) {
        stamp = now();
        if (cap_.retrieve(out) && !out.empty()) return true;
      }
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Frame grab/retrieve failed");
    }
    if (!fallback_frame_.empty()) {
      out = fallback_frame_;
      stamp = now();
      return true;
    }
    return false;
  }

  // ------------------------------------------------------------------
  // Timer callback
  // ------------------------------------------------------------------

  void timer_cb() {
    cv::Mat frame;
    builtin_interfaces::msg::Time frame_stamp;
    if (!get_frame(frame, frame_stamp)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "No frame available");
      return;
    }

    const auto t0 = std::chrono::steady_clock::now();

    // 1) person detection (RTMDet) -> bboxes (label 0 == person in COCO)
    std::vector<mmdeploy_rect_t> bboxes;
    std::vector<float> bbox_scores;
    mmdeploy::Detector::Result dets = detector_->Apply(frame);
    for (const mmdeploy_detection_t& det : dets) {
      if (det.label_id == 0 && det.score >= det_score_thr_) {
        bboxes.push_back(det.bbox);
        bbox_scores.push_back(det.score);
      }
    }

    // 2) pose estimation (RTMPose) on the person boxes
    person_pose_interfaces::msg::PersonPoseArray msg;
    // Audit F09 follow-up (8 Sep 2026): this used to be now(), taken here
    // AFTER detection+pose inference already ran - every consumer chained
    // off this stamp (person_body, then person_identity) was
    // aligning against a stamp that reflected processing-completion time,
    // not the frame's own actual capture time, silently biased by however
    // long inference took. frame_stamp (see get_frame()'s own comment) is
    // the frame's real capture time, or the closest available
    // approximation to it - never re-stamped by this node's own clock.
    msg.header.stamp = frame_stamp;
    msg.header.frame_id = frame_id_;
    cv::Mat vis = frame.clone();

    if (!bboxes.empty()) {
      mmdeploy::PoseDetector::Result poses = pose_detector_->Apply(frame, bboxes);

      // Pass 1: gate on keypoint coverage (see min_keypoints_'s own comment
      // - rejects e.g. a hand RTMDet mistook for "person").
      std::vector<size_t> candidates;
      for (size_t i = 0; i < poses.size(); ++i) {
        int n_confident = 0;
        for (int k = 0; k < poses[i].length; ++k) {
          if (poses[i].score[k] >= kp_score_thr_) ++n_confident;
        }
        if (n_confident < min_keypoints_) {
          RCLCPP_DEBUG(get_logger(),
                       "Rejected a detector box: only %d/%d keypoints >= %.2f (need %d) - "
                       "likely not a full body (e.g. a hand)",
                       n_confident, poses[i].length, kp_score_thr_, min_keypoints_);
          continue;
        }
        candidates.push_back(i);
      }

      // Pass 2: drop duplicate poses - RTMDet's own NMS operates on BOX iou,
      // which can miss two detections of the SAME physical person whose
      // boxes are cropped/scaled just differently enough to have low box
      // iou while their estimated joints land in nearly the same place
      // (confirmed as a real, live-observed failure mode). Skeleton overlap
      // is a more direct signal for "same person" than box overlap. Scale-
      // invariant threshold (a fraction of bbox diagonal) so this works the
      // same whether the person is close to or far from the camera. Keeps
      // whichever RTMDet itself was more confident about (bbox_score).
      std::vector<bool> dropped(candidates.size(), false);
      for (size_t ai = 0; ai < candidates.size(); ++ai) {
        if (dropped[ai]) continue;
        for (size_t bi = ai + 1; bi < candidates.size(); ++bi) {
          if (dropped[bi]) continue;
          size_t a = candidates[ai], b = candidates[bi];
          const mmdeploy_rect_t& box_a = bboxes[a];
          const mmdeploy_rect_t& box_b = bboxes[b];
          float diag_a = std::hypot(box_a.right - box_a.left, box_a.bottom - box_a.top);
          float diag_b = std::hypot(box_b.right - box_b.left, box_b.bottom - box_b.top);
          float dist_thr = 0.15f * 0.5f * (diag_a + diag_b);

          int shared = 0, close = 0;
          int n = std::min(poses[a].length, poses[b].length);
          for (int k = 0; k < n; ++k) {
            if (poses[a].score[k] < kp_score_thr_ || poses[b].score[k] < kp_score_thr_) continue;
            ++shared;
            float dx = poses[a].point[k].x - poses[b].point[k].x;
            float dy = poses[a].point[k].y - poses[b].point[k].y;
            if (std::hypot(dx, dy) < dist_thr) ++close;
          }
          // Need a meaningful number of shared confident keypoints to judge
          // by, and most of them close, to call these the same person.
          if (shared >= 4 && close >= (shared * 2) / 3) {
            bool drop_a = bbox_scores[a] < bbox_scores[b];
            dropped[drop_a ? ai : bi] = true;
            RCLCPP_DEBUG(get_logger(),
                         "Dropped a duplicate pose (%d/%d shared keypoints close) - "
                         "kept bbox_score=%.2f over %.2f",
                         close, shared, bbox_scores[drop_a ? b : a], bbox_scores[drop_a ? a : b]);
            if (drop_a) break;  // ai is gone - stop comparing it against the rest
          }
        }
      }

      // Pass 3: emit the survivors.
      for (size_t ci = 0; ci < candidates.size(); ++ci) {
        if (dropped[ci]) continue;
        size_t i = candidates[ci];
        const mmdeploy_pose_detection_t& pose = poses[i];

        person_pose_interfaces::msg::PersonPose person;
        const mmdeploy_rect_t& b = bboxes[i];
        person.bbox = {b.left, b.top, b.right, b.bottom};
        person.bbox_score = bbox_scores[i];
        person.keypoints.reserve(pose.length);
        for (int k = 0; k < pose.length; ++k) {
          person_pose_interfaces::msg::Keypoint2D kp;
          kp.x = pose.point[k].x;
          kp.y = pose.point[k].y;
          kp.score = pose.score[k];
          person.keypoints.push_back(kp);
        }
        msg.people.push_back(std::move(person));
        draw_bbox(vis, b, bbox_scores[i]);
        draw_pose(vis, pose);
      }
    }

    const double infer_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Inference FPS: %.1f",
                         infer_ms > 0 ? 1000.0 / infer_ms : 0.0);

    auto annot_msg = cv_bridge::CvImage(msg.header, "bgr8", vis).toImageMsg();
    annot_pub_->publish(*annot_msg);
    publisher_->publish(msg);

    // Cache for image_cb (input_source=="topic" only) - see its own comment
    // on why /pose/annotated is also republished directly off every raw
    // frame there, not just from this timer.
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      last_people_msg_ = msg;
    }
  }

  // Same overlay draw_bbox+draw_pose do, but reading a ROS message
  // (PersonKeypointsArray, a plain value type this node already owns a
  // copy of) instead of mmdeploy's own C-API result types - those are RAII
  // wrappers whose point/score arrays don't outlive the timer_cb call that
  // produced them, so they can't be cached for reuse by image_cb the way
  // this message can.
  void draw_annotations_from_msg(cv::Mat& img,
                                  const person_pose_interfaces::msg::PersonPoseArray& people) {
    for (const auto& person : people.people) {
      const cv::Point p1(static_cast<int>(person.bbox[0]), static_cast<int>(person.bbox[1]));
      const cv::Point p2(static_cast<int>(person.bbox[2]), static_cast<int>(person.bbox[3]));
      cv::rectangle(img, p1, p2, cv::Scalar(0, 200, 255), 2);
      char label[32];
      std::snprintf(label, sizeof(label), "person %.2f", person.bbox_score);
      cv::putText(img, label, cv::Point(p1.x, std::max(p1.y - 6, 12)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255), 2, cv::LINE_AA);

      const auto& kps = person.keypoints;
      for (const auto& kp : kps) {
        if (kp.score < kp_score_thr_) continue;
        cv::circle(img, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)), 3,
                   cv::Scalar(0, 255, 0), -1);
      }
      for (const auto& edge : kCocoSkeleton) {
        if (static_cast<size_t>(edge.first) >= kps.size() || static_cast<size_t>(edge.second) >= kps.size()) continue;
        if (kps[edge.first].score < kp_score_thr_ || kps[edge.second].score < kp_score_thr_) continue;
        cv::line(img,
                 cv::Point(static_cast<int>(kps[edge.first].x), static_cast<int>(kps[edge.first].y)),
                 cv::Point(static_cast<int>(kps[edge.second].x), static_cast<int>(kps[edge.second].y)),
                 cv::Scalar(255, 128, 0), 2);
      }
    }
  }

  // RTMDet's own accepted box + score - drawn separately from draw_pose so
  // a bad detector box (e.g. a hand RTMDet mistook for "person") is visible
  // on the preview even when the resulting skeleton is sparse/garbage -
  // RTMPose never rejects a box, it just estimates keypoints wherever it's
  // told to look (see timer_cb's own comment on this being pure top-down,
  // no pose-based veto of the detector's own boxes).
  void draw_bbox(cv::Mat& img, const mmdeploy_rect_t& box, float score) {
    const cv::Point p1(static_cast<int>(box.left), static_cast<int>(box.top));
    const cv::Point p2(static_cast<int>(box.right), static_cast<int>(box.bottom));
    cv::rectangle(img, p1, p2, cv::Scalar(0, 200, 255), 2);
    char label[32];
    std::snprintf(label, sizeof(label), "person %.2f", score);
    cv::putText(img, label, cv::Point(p1.x, std::max(p1.y - 6, 12)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
  }

  void draw_pose(cv::Mat& img, const mmdeploy_pose_detection_t& pose) {
    for (int i = 0; i < pose.length; ++i) {
      if (pose.score[i] < kp_score_thr_) continue;
      cv::circle(img, cv::Point(static_cast<int>(pose.point[i].x), static_cast<int>(pose.point[i].y)),
                 3, cv::Scalar(0, 255, 0), -1);
    }
    for (const auto& edge : kCocoSkeleton) {
      if (edge.first >= pose.length || edge.second >= pose.length) continue;
      if (pose.score[edge.first] < kp_score_thr_ || pose.score[edge.second] < kp_score_thr_) continue;
      cv::line(img,
               cv::Point(static_cast<int>(pose.point[edge.first].x),
                         static_cast<int>(pose.point[edge.first].y)),
               cv::Point(static_cast<int>(pose.point[edge.second].x),
                         static_cast<int>(pose.point[edge.second].y)),
               cv::Scalar(255, 128, 0), 2);
    }
  }

  // Members
  rclcpp::Publisher<person_pose_interfaces::msg::PersonPoseArray>::SharedPtr publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annot_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  cv::VideoCapture cap_;
  cv::Mat fallback_frame_;
  float det_score_thr_ = 0.5f;
  int min_keypoints_ = 5;
  float kp_score_thr_ = 0.3f;
  std::string frame_id_ = "camera";

  // Image-topic input source (e.g. D455)
  std::string input_source_ = "camera";
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  cv::Mat topic_frame_;
  builtin_interfaces::msg::Time topic_frame_stamp_;  // guarded by frame_mutex_ - the source image's own stamp
  person_pose_interfaces::msg::PersonPoseArray last_people_msg_;  // guarded by frame_mutex_
  std::mutex frame_mutex_;

  std::unique_ptr<mmdeploy::Detector> detector_;
  std::unique_ptr<mmdeploy::PoseDetector> pose_detector_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<PoseEstimationNode>());
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("person_pose"), "Fatal: %s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
