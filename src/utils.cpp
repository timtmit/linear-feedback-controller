#include "linear_feedback_controller/utils.hpp"

void remove_element(std::vector<std::string>* vector, std::string element)
{
    (*vector).erase(std::remove((*vector).begin(), (*vector).end(), element), (*vector).end());
}