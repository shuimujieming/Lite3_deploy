/**
 * @file lite3_test_policy_runner_onnx.cpp
 * @brief Implementation of Lite3TestPolicyRunnerONNX.
 *        This is the ONLY file that includes onnxruntime headers, so only
 *        this translation unit needs to be recompiled when policy logic changes.
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025  DeepRobotics
 */

#include "lite3_test_policy_runner_onnx.h"

#include <onnxruntime_cxx_api.h>

#include "basic_function.hpp"

#include <iostream>
#include <stdexcept>
#include <unordered_map>

// ---------------------------------------------------------------------------
// OrtImpl — all onnxruntime objects live here, invisible to callers
// ---------------------------------------------------------------------------
struct Lite3TestPolicyRunnerONNX::OrtImpl {
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::MemoryInfo memory_info;
    Ort::Session session;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    OrtImpl()
        : env(ORT_LOGGING_LEVEL_WARNING, "ONNXPolicy"),
          session_options(),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
          session(nullptr) {}
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
Lite3TestPolicyRunnerONNX::Lite3TestPolicyRunnerONNX(std::string policy_name)
    : PolicyRunnerBase(policy_name),
      ort_(std::make_unique<OrtImpl>()),
      gravity_direction(0., 0., -1.),
      current_obs_(single_obs_dim_),
      obs_history_(single_obs_dim_ * history_frames_),
      policy_input_(obs_dim_),
      joint_pos_rl(12),
      joint_vel_rl(12) {

    model_path_ = GetAbsPath() + "/../policy/ppo/policy.onnx";
    std::cout << "[ONNX INIT] Loading model: " << model_path_ << std::endl;

    ort_->session_options.SetIntraOpNumThreads(1);
    ort_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    ort_->session = Ort::Session(ort_->env, model_path_.c_str(), ort_->session_options);
    std::cout << "[ONNX INIT] Model loaded successfully.\n";

    const auto model_input_shape = ort_->session.GetInputTypeInfo(0)
                                       .GetTensorTypeAndShapeInfo()
                                       .GetShape();
    if (model_input_shape.size() != 2 ||
        (model_input_shape[0] != 1 && model_input_shape[0] != -1) ||
        model_input_shape[1] != obs_dim_) {
        const std::string actual_shape =
            model_input_shape.size() == 2
                ? "[" + std::to_string(model_input_shape[0]) + ", " +
                      std::to_string(model_input_shape[1]) + "]"
                : "rank " + std::to_string(model_input_shape.size());
        throw std::runtime_error(
            "ONNX input shape mismatch: expected [1, " +
            std::to_string(obs_dim_) + "] (or dynamic batch), got " + actual_shape);
    }

    ort_->input_names  = {"obs"};
    ort_->output_names = {"actions"};

    robot_order = {
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"};

    policy_order = {
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint"};

    action_scale_robot = {0.125f, 0.25f, 0.25f,
                          0.125f, 0.25f, 0.25f,
                          0.125f, 0.25f, 0.25f,
                          0.125f, 0.25f, 0.25f};

    dof_pos_default_policy.setZero(12);
    dof_pos_default_policy << 0.0000, -0.6500, 1.3000,
                                  0.0000, -0.6500, 1.3000,
                                  0.0000, -0.6500, 1.3000,
                                  0.0000, -0.6500, 1.3000;

    dof_pos_default_robot.setZero(12);
    dof_pos_default_robot << 0.0000, -0.6500, 1.3000,
                                  0.0000, -0.6500, 1.3000,
                                  0.0000, -0.6500, 1.3000,
                                  0.0000, -0.6500, 1.3000;


    kp_ = 30. * VecXf::Ones(12);
    kd_ =  1. * VecXf::Ones(12);
    max_cmd_vel_ << 2.0, 2.0, 0.8;

    current_obs_.setZero();
    obs_history_.setZero();
    policy_input_.setZero();
    last_action = VecXf::Zero(act_dim_);
    tmp_action = VecXf(act_dim_);
    ra.goal_joint_pos = VecXf::Zero(act_dim_);
    ra.goal_joint_vel = VecXf::Zero(act_dim_);
    ra.tau_ff         = VecXf::Zero(act_dim_);
    ra.kp = kp_;
    ra.kd = kd_;

    robot2policy_idx = generate_permutation(robot_order, policy_order);
    policy2robot_idx = generate_permutation(policy_order, robot_order);
    for (int i = 0; i < act_dim_; ++i) {
        std::cout << "robot2policy_idx[" << i << "]: " << robot2policy_idx[i] << std::endl;
        std::cout << "policy2robot_idx[" << i << "]: " << policy2robot_idx[i] << std::endl;
    }

    // Warm-up / sanity check
    for (int i = 0; i < 2; ++i) {
        VecXf dummy_input = VecXf::Ones(obs_dim_);
        std::array<int64_t, 2> input_shape{1, obs_dim_};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            ort_->memory_info, dummy_input.data(), obs_dim_,
            input_shape.data(), input_shape.size());

        auto reaction = ort_->session.Run(
            Ort::RunOptions{nullptr},
            ort_->input_names.data(), &input_tensor, 1,
            ort_->output_names.data(), 1);

        std::cout << policy_name_ << " ONNX policy network test success" << std::endl;
    }

    decimation_ = 12;
}

// The destructor must be defined in the .cpp where OrtImpl is complete.
Lite3TestPolicyRunnerONNX::~Lite3TestPolicyRunnerONNX() = default;

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------
void Lite3TestPolicyRunnerONNX::DisplayPolicyInfo() {
    std::cout << "ONNX policy: " << policy_name_ << "\n";
    std::cout << "path: " << model_path_ << "\n";
    std::cout << "single_obs_dim: " << single_obs_dim_
              << ", history_frames: " << history_frames_
              << ", input_dim: " << obs_dim_
              << ", action_dim: " << act_dim_ << "\n";
}

void Lite3TestPolicyRunnerONNX::OnEnter() {
    run_cnt_ = 0;
    current_obs_.setZero();
    obs_history_.setZero();
    policy_input_.setZero();
    last_action.setZero();
    std::cout << "[ONNX ENTER] PolicyRunner entered: " << policy_name_ << std::endl;
}

RobotAction Lite3TestPolicyRunnerONNX::GetRobotAction(const RobotBasicState& ro) {
    Vec3f base_omega       = ro.base_omega * omega_scale_;
    Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction;
    Vec3f cmd_vel          = ro.cmd_vel_normlized.cwiseProduct(max_cmd_vel_);

    for (int i = 0; i < act_dim_; ++i) {
        joint_pos_rl(i) = ro.joint_pos(robot2policy_idx[i]);
        joint_vel_rl(i) = ro.joint_vel(robot2policy_idx[i]) * dof_vel_scale_;
    }
    joint_pos_rl -= dof_pos_default_policy;

    current_obs_.setZero();
    current_obs_ << base_omega,
                    projected_gravity,
                    cmd_vel,
                    joint_pos_rl,
                    joint_vel_rl,
                    last_action;

    // Match the training-side layout:
    // torch.cat([obs_buf, obs_history_buf.view(num_envs, -1)], dim=-1).
    // The history buffer is chronological (oldest -> newest), so the input is
    // [current, oldest_history, ..., newest_history].
    policy_input_.head(single_obs_dim_) = current_obs_;
    policy_input_.tail(obs_history_.size()) = obs_history_;

    std::array<int64_t, 2> input_shape{1, obs_dim_};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        ort_->memory_info,
        policy_input_.data(), policy_input_.size(),
        input_shape.data(), input_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(input_tensor));

    auto output_tensors = ort_->session.Run(
        Ort::RunOptions{nullptr},
        ort_->input_names.data(), inputs.data(), 1,
        ort_->output_names.data(), 1);

    float* action_data = output_tensors[0].GetTensorMutableData<float>();
    Eigen::Map<Eigen::MatrixXf> act(action_data, act_dim_, 1);
    action      = VecXf(act);
    last_action = action;

    // Match:
    //   episode_length <= 1: stack([obs_buf] * history_len)
    //   otherwise: cat([history[:, 1:], obs_buf.unsqueeze(1)], dim=1)
    if (run_cnt_ <= 1) {
        for (int frame = 0; frame < history_frames_; ++frame) {
            obs_history_.segment(frame * single_obs_dim_, single_obs_dim_) =
                current_obs_;
        }
    } else {
        for (int frame = 0; frame < history_frames_ - 1; ++frame) {
            obs_history_.segment(frame * single_obs_dim_, single_obs_dim_) =
                obs_history_.segment((frame + 1) * single_obs_dim_, single_obs_dim_);
        }
        obs_history_.tail(single_obs_dim_) = current_obs_;
    }

    for (int i = 0; i < act_dim_; ++i) {
        tmp_action(i)  = action(policy2robot_idx[i]);
        tmp_action(i) *= action_scale_robot[i];
    }
    tmp_action += dof_pos_default_robot;

    ra.goal_joint_pos = tmp_action;
    ra.goal_joint_vel = VecXf::Zero(act_dim_);
    ra.tau_ff         = VecXf::Zero(act_dim_);
    ra.kp = kp_;
    ra.kd = kd_;
    ++run_cnt_;

    return ra;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------
std::vector<int> Lite3TestPolicyRunnerONNX::generate_permutation(
    const std::vector<std::string>& from,
    const std::vector<std::string>& to,
    int default_index)
{
    std::unordered_map<std::string, int> idx_map;
    for (int i = 0; i < (int)from.size(); ++i)
        idx_map[from[i]] = i;

    std::vector<int> perm;
    for (const auto& name : to) {
        auto it = idx_map.find(name);
        perm.push_back(it != idx_map.end() ? it->second : default_index);
    }
    return perm;
}
