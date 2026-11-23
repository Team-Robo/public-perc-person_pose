"""Launch the C++ pose estimation node.

`device` drives everything — the MMDeploy SDK, the runtime libs injected into
LD_LIBRARY_PATH, and the default model directory:

    device:=cuda  → CUDA/TensorRT SDK + TensorRT/cuDNN libs + ~/models/rtmdet_trt|rtmpose_trt
    device:=cpu   → CPU/ONNXRuntime SDK                      + ~/models/rtmdet|rtmpose

so `ros2 launch person_pose person_pose.launch.py` just works, with no
shell env setup. Override anything on the CLI, e.g.:

    ros2 launch person_pose person_pose.launch.py device:=cpu input_source:=camera
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HOME = os.path.expanduser('~')


def _dir(env_name, default):
    """Env override, but only if it points at a real dir (ignores stale env vars)."""
    p = os.environ.get(env_name)
    return p if (p and os.path.isdir(p)) else default


# SDKs and out-of-tree runtime libs (the node dlopen's these at runtime).
POSE_ENV_DIR = _dir('POSE_ENV_DIR', os.path.join(HOME, 'pose-env'))
MMDEPLOY_CPU = _dir('MMDEPLOY_HOME', os.path.join(HOME, 'mmdeploy-1.3.1-linux-x86_64-cxx11abi'))
MMDEPLOY_CUDA = _dir('MMDEPLOY_CUDA_HOME',
                     os.path.join(HOME, 'mmdeploy-1.3.1-linux-x86_64-cxx11abi-cuda11.8'))
ORT_LIB = os.path.join(_dir('ONNXRUNTIME_DIR',
                            os.path.join(HOME, 'onnxruntime-linux-x64-1.15.1')), 'lib')
TRT_LIB = _dir('TENSORRT_LIB',
               os.path.join(POSE_ENV_DIR, 'lib/python3.10/site-packages/tensorrt_libs'))
CUDNN_LIB = _dir('CUDNN_LIB',
                 os.path.join(POSE_ENV_DIR, 'lib/python3.10/site-packages/nvidia/cudnn/lib'))


def _launch_setup(context, *_args, **_kwargs):
    device = LaunchConfiguration('device').perform(context)
    is_cuda = device == 'cuda'
    # Precision selects which prebuilt TensorRT engine dir to load (fp16 vs int8).
    # It does NOT re-quantize — precision is baked into the .engine at export time.
    precision = LaunchConfiguration('precision').perform(context)  # 'fp16' | 'int8'
    trt_suffix = '_trt_int8' if precision == 'int8' else '_trt'

    # Model dir default depends on device + precision (empty CLI value → pick default).
    det = LaunchConfiguration('det_model_path').perform(context)
    pose = LaunchConfiguration('pose_model_path').perform(context)
    if not det:
        det = os.path.join(HOME, 'models', ('rtmdet' + trt_suffix) if is_cuda else 'rtmdet')
    if not pose:
        pose = os.path.join(HOME, 'models', ('rtmpose' + trt_suffix) if is_cuda else 'rtmpose')

    warnings = []
    for role, path in (('det', det), ('pose', pose)):
        if not os.path.isdir(path):
            warnings.append(LogInfo(msg=(
                f"[person_pose_python] {role}_model_path '{path}' does not exist. "
                f"For device:={device} precision:={precision} you must export that engine "
                "first (see README §A-GPU / §B for INT8 calibration)." )))

    if is_cuda:
        lib_dirs = [os.path.join(MMDEPLOY_CUDA, 'lib'), TRT_LIB, CUDNN_LIB, ORT_LIB]
    else:
        lib_dirs = [os.path.join(MMDEPLOY_CPU, 'lib'), ORT_LIB]
    ld_library_path = os.pathsep.join(lib_dirs + [os.environ.get('LD_LIBRARY_PATH', '')])

    node = Node(
        package='person_pose',
        executable='person_pose',
        output='screen',
        additional_env={'LD_LIBRARY_PATH': ld_library_path},
        parameters=[{
            'camera_index': LaunchConfiguration('camera_index'),
            'use_fallback': LaunchConfiguration('use_fallback'),
            'fallback_image_path': LaunchConfiguration('fallback_image_path'),
            'target_fps': LaunchConfiguration('target_fps'),
            'input_source': LaunchConfiguration('input_source'),
            'image_topic': LaunchConfiguration('image_topic'),
            'det_model_path': det,
            'pose_model_path': pose,
            'device': device,
            'det_score_thr': LaunchConfiguration('det_score_thr'),
            # keypoint filter, exposed so callers (seating_detection's system.launch.py,
            # dashboard targets) can tune it
            'min_keypoints': LaunchConfiguration('min_keypoints'),
            'keypoint_score_thr': LaunchConfiguration('keypoint_score_thr'),
            'frame_id': LaunchConfiguration('frame_id'),
        }],
    )
    return warnings + [node]


def generate_launch_description():
    args = [
        DeclareLaunchArgument('device', default_value='cuda'),   # 'cuda' (TensorRT) or 'cpu' (ORT)
        # TensorRT engine precision to load (cuda only): 'fp16' (default) or 'int8'.
        # Selects ~/models/rtmdet_trt vs ~/models/rtmdet_trt_int8 — the int8 engines must
        # be exported with calibration first (README). Does not re-quantize at runtime.
        DeclareLaunchArgument('precision', default_value='fp16'),
        DeclareLaunchArgument('camera_index', default_value='0'),
        DeclareLaunchArgument('use_fallback', default_value='false'),
        DeclareLaunchArgument('fallback_image_path',
                              default_value=os.path.join(HOME, 'test_pose.jpg')),
        DeclareLaunchArgument('target_fps', default_value='30.0'),
        # 'camera' = direct USB capture; 'topic' = subscribe to image_topic (e.g. D455)
        DeclareLaunchArgument('input_source', default_value='camera'),
        DeclareLaunchArgument('image_topic', default_value='/camera/color/image_raw'),
        # empty → per-device default (rtmdet_trt/rtmpose_trt for cuda, rtmdet/rtmpose for cpu)
        DeclareLaunchArgument('det_model_path', default_value=''),
        DeclareLaunchArgument('pose_model_path', default_value=''),
        DeclareLaunchArgument('det_score_thr', default_value='0.3'),
        # a person is kept only if >= min_keypoints of 17 keypoints score >= keypoint_score_thr
        # (stock node defaults: 5 and 0.30; loosened for partly-occluded people)
        DeclareLaunchArgument('min_keypoints', default_value='4'),
        DeclareLaunchArgument('keypoint_score_thr', default_value='0.25'),
        DeclareLaunchArgument('frame_id', default_value='camera'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch_setup)])
