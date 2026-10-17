/**
 * @file rl_control_state_onnx.hpp
 * @brief rl policy runnning state using onnx
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-08-10
 * 
 * @copyright Copyright (c) 2025  DeepRobotics
 * 
 */




#pragma once

#include "state_base.h"
#include "policy_runner_base.hpp"
#include "lite3_test_policy_runner_onnx.h"

#include <atomic>
#include <mutex>



class RLControlStateONNX : public StateBase
{
private:
    RobotBasicState rbs_;
    std::shared_ptr<PolicyRunnerBase> policy_ptr_;


    
    std::thread run_policy_thread_;
    std::atomic<bool> start_flag_{false};
    std::atomic<int> state_run_cnt_{-1};
    mutable std::mutex observation_mutex_;
    Vec3f filtered_nav_command_ = Vec3f::Zero();
    const Vec3f manual_max_cmd_vel_ = Vec3f(0.5f, 0.5f, 0.8f);
    const Vec3f nav_cmd_min_ = Vec3f(-0.5f, -0.5f, -0.8f);
    const Vec3f nav_cmd_max_ = Vec3f(0.5f, 0.5f, 0.8f);
    const float command_filter_alpha_ = 0.5f;
    bool navigation_mode_ = false;  // Changed only after the policy thread stops.

    float policy_cost_time_ = 1;

    void UpdateRobotObservation(){
        std::lock_guard<std::mutex> observation_lock(observation_mutex_);
        rbs_.base_rpy     = ri_ptr_->GetImuRpy();
        rbs_.base_rot_mat = RpyToRm(rbs_.base_rpy);
        rbs_.projected_gravity = RmToProjectedGravity(rbs_.base_rot_mat);
        rbs_.base_omega   = ri_ptr_->GetImuOmega();
        rbs_.base_acc     = ri_ptr_->GetImuAcc();
        rbs_.joint_pos    = ri_ptr_->GetJointPosition();
        rbs_.joint_vel    = ri_ptr_->GetJointVelocity();
        rbs_.joint_tau    = ri_ptr_->GetJointTorque();
        // static Vec3f cmd_vel;
        // Vec3f cmd_vel_input = Vec3f(uc_ptr_->GetUserCommand().forward_vel_scale, 
        //                             uc_ptr_->GetUserCommand().side_vel_scale, 
        //                             uc_ptr_->GetUserCommand().turnning_vel_scale);

        // Eigen::Vector3f vel_delta = cmd_vel_input - cmd_vel;
        // const Eigen::Vector3f vel_delta_const(0.0015, 0.001, 0.0012);
        // for(int i=0;i<3;++i){
        //     if(fabs(vel_delta(i)) > vel_delta_const(i)) vel_delta(i) = Sign(vel_delta(i))*vel_delta_const(i);
        // }
        // cmd_vel+=vel_delta;           
        // rbs_.cmd_vel_normlized = cmd_vel;
        const auto user_command = uc_ptr_->GetUserCommand();
        const Vec3f manual_command(user_command.forward_vel_scale,
                                    user_command.side_vel_scale,
                                    user_command.turnning_vel_scale);
        rbs_.cmd_vel_normlized = navigation_mode_ ? Vec3f::Zero() : manual_command;

        Vec3f requested_command = manual_command.cwiseProduct(manual_max_cmd_vel_);
        if (navigation_mode_) {
            requested_command.setZero();
            if (sea_nav_ptr_) {
                const auto tcp_command = sea_nav_ptr_->GetCommand();
                // Navigation never falls back to joystick velocity.
                requested_command =
                    (tcp_command.received && tcp_command.fresh && tcp_command.enable &&
                     !tcp_command.emergency_stop)
                        ? tcp_command.velocity
                        : Vec3f::Zero();
            }
        }
        requested_command = requested_command.cwiseMax(nav_cmd_min_).cwiseMin(nav_cmd_max_);
        filtered_nav_command_ = command_filter_alpha_ * requested_command +
                                (1.0f - command_filter_alpha_) * filtered_nav_command_;
        rbs_.cmd_vel_physical = filtered_nav_command_;
        rbs_.cmd_vel_physical_valid = true;
        if (sea_nav_ptr_) sea_nav_ptr_->SetAppliedNavCommand(filtered_nav_command_);
        
    }

    void PolicyRunner(){
        int run_cnt_record = -1;
        while (start_flag_.load(std::memory_order_acquire)){
            const int state_run_cnt = state_run_cnt_.load(std::memory_order_acquire);
            
            if(state_run_cnt >= 0 && state_run_cnt%policy_ptr_->decimation_ == 0 &&
               state_run_cnt != run_cnt_record){
                timespec start_timestamp, end_timestamp;
                clock_gettime(CLOCK_MONOTONIC,&start_timestamp);
                RobotBasicState observation;
                {
                    std::lock_guard<std::mutex> observation_lock(observation_mutex_);
                    observation = rbs_;
                }
                auto ra = policy_ptr_->GetRobotAction(observation);
                MatXf res = ra.ConvertToMat();
                ri_ptr_->SetJointCommand(res); // (current torque, not last torque, video content slip of the tongue)
                run_cnt_record = state_run_cnt;
                clock_gettime(CLOCK_MONOTONIC,&end_timestamp);
                policy_cost_time_ = (end_timestamp.tv_sec-start_timestamp.tv_sec)*1e3 
                                    +(end_timestamp.tv_nsec-start_timestamp.tv_nsec)/1e6;
                // std::cout << "cost_time:  " << policy_cost_time_ << " ms\n";
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

public:
    RLControlStateONNX(const RobotType& robot_type, const std::string& state_name, 
        std::shared_ptr<ControllerData> data_ptr,
        std::shared_ptr<PolicyRunnerBase> policy = nullptr):StateBase(robot_type, state_name, data_ptr){
        rbs_.cmd_vel_normlized = Vec3f::Zero();
        rbs_.cmd_vel_physical = Vec3f::Zero();
        rbs_.cmd_vel_physical_valid = false;
        policy_ptr_ = policy ? std::move(policy)
                            : std::make_shared<Lite3TestPolicyRunnerONNX>("test_onnx");
        if(!policy_ptr_){
            std::cerr << "[ERROR] Failed to initialize ONNX policy runner." << std::endl;
            exit(0);
        }  
        policy_ptr_->DisplayPolicyInfo();
        }
    ~RLControlStateONNX(){}

    void SetNavigationMode(bool enabled) {
        navigation_mode_ = enabled;
        state_name_ = enabled ? "navigation" : "rl_control";
    }

    virtual void OnEnter() {
        if (navigation_mode_ && sea_nav_ptr_) sea_nav_ptr_->ClearCommand();
        policy_ptr_->OnEnter();
        {
            std::lock_guard<std::mutex> observation_lock(observation_mutex_);
            filtered_nav_command_.setZero();
        }
        state_run_cnt_.store(-1, std::memory_order_release);
        start_flag_.store(true, std::memory_order_release);
        run_policy_thread_ = std::thread(std::bind(&RLControlStateONNX::PolicyRunner, this));
        StateBase::msfb_.UpdateCurrentState(navigation_mode_
            ? RobotMotionState::NavigationMode : RobotMotionState::RLControlMode);
        uc_ptr_->SetMotionStateFeedback(StateBase::msfb_);
    };

    virtual void OnExit() { 
        start_flag_.store(false, std::memory_order_release);
        if (run_policy_thread_.joinable()) run_policy_thread_.join();
        state_run_cnt_.store(-1, std::memory_order_release);
        if (sea_nav_ptr_) sea_nav_ptr_->SetAppliedNavCommand(Vec3f::Zero());
    }

    virtual void Run() {
        UpdateRobotObservation();
        ds_ptr_->InsertScopeData(0, policy_cost_time_);
        state_run_cnt_.fetch_add(1, std::memory_order_release);
    }

    virtual bool LoseControlJudge() {
        if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::JointDamping)) return true;
        if(navigation_mode_ && sea_nav_ptr_ && sea_nav_ptr_->IsEmergencyStopActive()) return true;
        return PostureUnsafeCheck();
    }

    bool PostureUnsafeCheck(){
        Vec3f rpy = ri_ptr_->GetImuRpy();
        if(fabs(rpy(0)) > 30./180*M_PI || fabs(rpy(1)) > 45./180*M_PI){
            std::cout << "posture value: " << 180./M_PI*rpy.transpose() << std::endl;
            return true;
        }
        return false;
    }

    virtual StateName GetNextStateName() {
        if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::LieDown)) return StateName::kLieDown;
        if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::NavigationMode)) return StateName::kNavigation;
        if(uc_ptr_->GetUserCommand().target_mode == int(RobotMotionState::RLControlMode)) return StateName::kRLControl;
        return navigation_mode_ ? StateName::kNavigation : StateName::kRLControl;
    }
};
