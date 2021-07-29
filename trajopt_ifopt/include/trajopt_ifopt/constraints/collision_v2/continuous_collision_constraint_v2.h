/**
 * @file continuous_collision_constraint.h
 * @brief The continuous collision position constraint
 *
 * @author Levi Armstrong
 * @author Matthew Powelson
 * @date May 18, 2020
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2020, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRAJOPT_IFOPT_CONTINUOUS_COLLISION_CONSTRAINT_V2_H
#define TRAJOPT_IFOPT_CONTINUOUS_COLLISION_CONSTRAINT_V2_H
#include <trajopt_utils/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <Eigen/Eigen>
#include <ifopt/constraint_set.h>
#include <tesseract_collision/core/common.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt_ifopt/constraints/collision/continuous_collision_evaluators.h>
#include <trajopt_ifopt/constraints/collision/continuous_combine_collision_data.h>
#include <trajopt_ifopt/variable_sets/joint_position_variable.h>

namespace trajopt_ifopt
{
class ContinuousCollisionConstraintIfoptV2 : public ifopt::ConstraintSet
{
public:
  using Ptr = std::shared_ptr<ContinuousCollisionConstraintIfoptV2>;
  using ConstPtr = std::shared_ptr<const ContinuousCollisionConstraintIfoptV2>;

  /**
   * @brief ContinuousCollisionConstraintIfopt
   * @param collision_evaluator The continuous collision evaluator to use
   * @param combine_methods The methods for combining collision data into a single error and jacobian
   * @param position_vars The position vars associated with this constraint. The middle position var is the var set for
   * the constraint. The pre and post are supporting variables for calculating the constraint error and gradient
   *
   *  There are three conditions of the array that are acceptable:
   *    Case 1: <nullptr, ConstPtr, ConstPtr> This would be usually be the first position in the trajectory where it
   * does not have previous position var set Case 2: <ConstPtr, ConstPtr, nullptr> This would be usually be the last
   * position in the trajectory where it does not have post position var set Case 3: <ConstPtr, ConstPtr, ConstPtr> This
   * would usually be used for all other states in the trajectory because it has both a pre and post position var set
   * @param name
   */
  ContinuousCollisionConstraintIfoptV2(ContinuousCollisionEvaluator::Ptr collision_evaluator,
                                       std::array<JointPosition::ConstPtr, 2> position_vars,
                                       std::array<bool, 2> position_vars_fixed,
                                       int max_num_cnt,
                                       const std::vector<tesseract_collision::ObjectPairKey>& collision_object_pairs,
                                       const std::string& name = "LVSCollision");

  /**
   * @brief Returns the values associated with the constraint.
   * @warning Make sure that the values returns are not just the violation but the constraint values.
   * Remember the values are the constant in the quadratic function, so if you only return the
   * violation then if it is not violating the constraint this would be zero which means it
   * will always appear to be on the constraint boundary which will cause issue when solving.
   * If it is not voliating the constraint then return the max negative number.
   * If no contacts are found return the negative of the collision margin buffer. This is why
   * it is important to not set the collision margin buffer to zero.
   * @return The constraint values not the violations
   */
  Eigen::VectorXd GetValues() const override;

  /**
   * @brief  Returns the "bounds" of this constraint. How these are enforced is up to the solver
   * @return Returns the "bounds" of this constraint
   */
  std::vector<ifopt::Bounds> GetBounds() const override;
  /**
   * @brief Fills the jacobian block associated with the given var_set.
   * @param var_set Name of the var_set to which the jac_block is associated
   * @param jac_block Block of the overall jacobian associated with these constraints and the var_set variable
   */
  void FillJacobianBlock(std::string var_set, Jacobian& jac_block) const override;

  /**
   * @brief Sets the bounds on the collision distance
   * @param bounds New bounds that will be set. Should be size 1
   */
  void SetBounds(const std::vector<ifopt::Bounds>& bounds);

  /**
   * @brief Get the collision evaluator. This exposed for plotter callbacks
   * @return The collision evaluator
   */
  ContinuousCollisionEvaluator::Ptr GetCollisionEvaluator() const;

private:
  /** @brief The number of joints in a single JointPosition */
  long n_dof_;

  /** @brief Bounds on the constraint value. Default: std::vector<Bounds>(1, ifopt::BoundSmallerZero) */
  std::vector<ifopt::Bounds> bounds_;

  /**
   * @brief Pointers to the vars used by this constraint.
   * Do not access them directly. Instead use this->GetVariables()->GetComponent(position_var->GetName())->GetValues()
   */
  std::array<JointPosition::ConstPtr, 2> position_vars_;
  std::array<bool, 2> position_vars_fixed_;

  ContinuousCollisionEvaluator::Ptr collision_evaluator_;
};
};  // namespace trajopt_ifopt

#endif  // TRAJOPT_IFOPT_CONTINUOUS_COLLISION_CONSTRAINT_H
