#include "linear_feedback_controller/linear_feedback_controller.hpp"

namespace linear_feedback_controller {

LinearFeedbackController::LinearFeedbackController() {
  robot_model_builder_ = std::make_shared<RobotModelBuilder>();
  control_.resize(0);
  first_control_received_time_ = TimePoint::min();
}

LinearFeedbackController::~LinearFeedbackController() {}

bool LinearFeedbackController::load(const ControllerParameters& params) {
  params_ = params;

  // Load the robot model.
  if (!robot_model_builder_->build_model(
          params_.urdf, params_.moving_joint_names,
          params_.controlled_joint_names, params_.robot_has_free_flyer)) {
    return false;
  }

  // Setup the pd controller.
  pd_controller_.initialize(robot_model_builder_);
  pd_controller_.set_gains(params_.p_gains, params_.d_gains);

  // Setup the lfc controller.
  lf_controller_.initialize(robot_model_builder_);

  // Allocate memory
  robot_configuration_ = Eigen::VectorXd::Zero(robot_model_builder_->get_nq());
  robot_velocity_ = Eigen::VectorXd::Zero(robot_model_builder_->get_nv());
  robot_velocity_null_ = Eigen::VectorXd::Zero(robot_model_builder_->get_nv());

  tau_init_ = Eigen::VectorXd::Zero(robot_model_builder_->get_joint_nv());
  tau_gravity_ = Eigen::VectorXd::Zero(robot_model_builder_->get_joint_nv());
  control_pd_ = Eigen::VectorXd::Zero(robot_model_builder_->get_joint_nv());
  control_lf_ = Eigen::VectorXd::Zero(robot_model_builder_->get_joint_nv());
  control_ = Eigen::VectorXd::Zero(robot_model_builder_->get_joint_nv());
  integrated_position_ =
      Eigen::VectorXd::Zero(robot_model_builder_->get_joint_nv());
  integrated_velocity_ =
      Eigen::VectorXd::Zero(robot_model_builder_->get_joint_nv());

  return true;
}

bool LinearFeedbackController::set_initial_state(
    const Eigen::VectorXd& tau_init, const Eigen::VectorXd& jq_init) {
  const int joint_nq = robot_model_builder_->get_joint_hw_nq();
  const int joint_nv = robot_model_builder_->get_joint_nv();

  // verification of tau_init size
  if (tau_init.size() != joint_nv) {
    std::stringstream ss;
    ss << "[LFC] set_initial_state: tau_init size=" << tau_init.size()
       << " but expected " << joint_nv;
    std::cerr << ss.str() << std::endl;
    return false;
  }

  // verification of jq_init size
  if (jq_init.size() != joint_nq) {
    std::stringstream ss;
    ss << "[LFC] set_initial_state: jq_init size= " << jq_init.size()
       << " but expected " << joint_nq;
    std::cerr << ss.str() << std::endl;
    return false;
  }

  pd_controller_.set_reference(tau_init, jq_init);
  tau_init_ = tau_init;
  integrated_position_.tail(jq_init.size())= jq_init;
  return true;
}

const Eigen::VectorXd& LinearFeedbackController::compute_control(
    const TimePoint& time, const Sensor& sensor, const Control& control,
    const bool remove_gravity_compensation_effort) {
  // Shortcuts for easier code writing.
  const auto& sensor_js = sensor.joint_state;
  const auto& ctrl_js = control.initial_state.joint_state;
  const TimePoint t0 = TimePoint(
      std::chrono::nanoseconds(control.initial_state.stamp.nanoseconds()));

  // Self documented variables.
  const bool control_msg_received = !control.feedforward.effort.hasNaN();
  const bool first_control_received_time_initialized =
      first_control_received_time_ != TimePoint::min();
  const bool during_switch = (time - first_control_received_time_) <
                             params_.pd_to_lf_transition_duration;

  // Check whenever the first data has arrived and save the time.
  if (control_msg_received && !first_control_received_time_initialized) {
    first_control_received_time_ = time;
  }

  if (remove_gravity_compensation_effort) {
    robot_model_builder_->construct_robot_state(sensor, robot_configuration_,
                                                robot_velocity_);

    // NOTE: .tail() is used to remove the freeflyer components
    tau_gravity_ =
        pinocchio::rnea(robot_model_builder_->get_model(),
                        robot_model_builder_->get_data(), robot_configuration_,
                        robot_velocity_null_, robot_velocity_null_)
            .tail(tau_init_.size());
  }

  // inputs for PD controller
  const int joint_hw_nq = robot_model_builder_->get_joint_hw_nq();
  const int joint_nv = robot_model_builder_->get_joint_nv();

  // Sanity check design : PD / hardware are using joint_hw_nq == joint_nv
  if (joint_hw_nq != joint_nv) {
    std::stringstream ss;
    ss << "[LFC] compute_control: design error, joint_hw_nq=" << joint_hw_nq
       << " but joint_nv=" << joint_nv;
    std::cerr << ss.str() << std::endl;
    throw std::logic_error(ss.str());
  }

  // Verification on the sizes coming from ROS
  if (sensor_js.position.size() != joint_hw_nq) {
    std::stringstream ss;
    ss << "[LFC] compute_control: unexpected sensor_js.position size="
       << sensor_js.position.size() << " (expected " << joint_hw_nq
       << " = joint_position_nq)";
    std::cerr << ss.str() << std::endl;
    throw std::invalid_argument(ss.str());
  }

  if (sensor_js.velocity.size() != joint_nv) {
    std::stringstream ss;
    ss << "[LFC] compute_control: unexpected sensor_js.velocity size="
       << sensor_js.velocity.size() << " (expected " << joint_nv
       << " = joint_nv)";
    std::cerr << ss.str() << std::endl;
    throw std::invalid_argument(ss.str());
  }

  // PD Control Phase
  if (!first_control_received_time_initialized) {
    control_pd_ = pd_controller_.compute_control(sensor_js.position, sensor_js.velocity);
    if (remove_gravity_compensation_effort) control_pd_ -= tau_init_;

    // Effort joints : PD
    for (int i : params_.joint_effort_idx) control_(i) = control_pd_(i);
    // Position joints : keep same position
    for (int i : params_.joint_position_idx) control_(i) = sensor_js.position(i);
    // Velocity joints : zero or same speed
    for (int i : params_.joint_velocity_idx) control_(i) = 0.0;

    return control_;
  }

  // Integration
  const int nq = robot_model_builder_->get_nq();
  const int nv = robot_model_builder_->get_nv();

  if (control.feedforward.position.size() != nq) {
    std::stringstream ss;
    ss << "[LFC] compute_control: unexpected control.feedforward.position "
          "size="
       << control.feedforward.position.size() << " (expected " << nq
       << " = nq)";
    std::cerr << ss.str() << std::endl;
    throw std::invalid_argument(ss.str());
  }
  if (control.feedforward.velocity.size() != nv) {
    std::stringstream ss;
    ss << "[LFC] compute_control: unexpected control.feedforward.velocity "
          "size="
       << control.feedforward.velocity.size() << " (expected " << nv
       << " = nv)";
    std::cerr << ss.str() << std::endl;
    throw std::invalid_argument(ss.str());
  }
  if (control.feedforward.acceleration.size() != nv) {
    std::stringstream ss;
    ss << "[LFC] compute_control: unexpected "
          "control.feedforward.acceleration size="
       << control.feedforward.acceleration.size() << " (expected " << nv
       << " = nv)";
    std::cerr << ss.str() << std::endl;
    throw std::invalid_argument(ss.str());
  }

  double delta_t = std::chrono::duration<double>(time - t0).count();
  // v(t) = dq*(t0) + ddq*(t0) * delta_t
  // integrated_velocity_ += control.feedforward.acceleration * delta_t;
  integrated_velocity_ = control.feedforward.velocity + control.feedforward.acceleration * delta_t;

  // integrated_position_ = pinocchio::integrate(robot_model_builder_->get_model(), integrated_position_, control.feedforward.velocity*delta_t);
  // q(t) = q0 + v0 * delta_t + 0.5 * a0 * delta_t^2
  // integrated_position_ = pinocchio::integrate(
  //     robot_model_builder_->get_model(), 
  //     integrated_position_, 
  //     integrated_velocity_ + 0.5 * control.feedforward.acceleration * delta_t * delta_t
  // );
  Eigen::VectorXd displacement = control.feedforward.velocity * delta_t + 
                               0.5 * control.feedforward.acceleration * delta_t * delta_t;

  integrated_position_ = pinocchio::integrate(
      robot_model_builder_->get_model(), 
      control.feedforward.position,
      displacement
  );

  const auto& model = robot_model_builder_->get_model();
  const auto& moving_names = robot_model_builder_->get_moving_joint_names();
  const auto& pin_to_hwi =
      robot_model_builder_->get_pinocchio_to_hardware_interface_map();

  Eigen::VectorXd hw_position = Eigen::VectorXd::Zero(joint_hw_nq);
  Eigen::VectorXd hw_velocity = Eigen::VectorXd::Zero(joint_hw_nq);

  for (std::size_t k = 0; k < moving_names.size(); ++k) {
    const auto joint_id = model.getJointId(moving_names[k]);
    const auto& jmodel = model.joints[joint_id];
    const int idx_q = jmodel.idx_q();
    const int idx_v = jmodel.idx_v();
    const int nq_k = jmodel.nq();
    const int hw_idx = pin_to_hwi.at(static_cast<int>(k));

    if (nq_k == 1) {
      // Joint 1DOF
      hw_position(hw_idx) = integrated_position_(idx_q);
    } else if (nq_k == 2) {
      // Joint SO(2) : q = (cos theta, sin theta)
      hw_position(hw_idx) =
          std::atan2(integrated_position_(idx_q + 1), integrated_position_(idx_q));
    } else {
      std::stringstream ss;
      ss << "[LFC] compute_control: unsupported nq=" << nq_k
        << " for joint '" << moving_names[k] << "' during hardware position extraction";
      throw std::logic_error(ss.str());
    }

    hw_velocity(hw_idx) = integrated_velocity_(idx_v);
  }
  // Switching Phase (PD -> LF)
  if (during_switch) {
    double weight = (double)((time - first_control_received_time_).count()) /
                  (double)params_.pd_to_lf_transition_duration.count();
    weight = std::clamp(weight, 0.0, 1.0);
    
    control_pd_ = pd_controller_.compute_control(sensor_js.position, sensor_js.velocity);
    control_lf_ = lf_controller_.compute_control(sensor, control);

    if (remove_gravity_compensation_effort) {
      control_pd_ -= tau_init_;
      control_lf_ -= tau_gravity_;
    }
      for (int i : params_.joint_effort_idx) {
        control_(i) = (1.0 - weight) * control_pd_(i) + weight * control_lf_(i);
      }
    for (int i : params_.joint_position_idx) control_(i) = hw_position(i);
    for (int i : params_.joint_velocity_idx) control_(i) = hw_velocity(i);
    return control_;
  }

  // LF Control Phase
  control_lf_ = lf_controller_.compute_control(sensor, control);
  if (remove_gravity_compensation_effort) control_lf_ -= tau_gravity_;
  // Commands routing
  for (int i : params_.joint_effort_idx)   control_(i) = control_lf_(i);
  for (int i : params_.joint_position_idx) control_(i) = hw_position(i);
  for (int i : params_.joint_velocity_idx) control_(i) = hw_velocity(i);

  return control_;
}

RobotModelBuilder::ConstSharedPtr LinearFeedbackController::get_robot_model()
    const {
  return robot_model_builder_;
}

}  // namespace linear_feedback_controller