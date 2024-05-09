% Define which section teaches which concept
teaches(introduction, computer_programs_and_business_logic).
teaches(introduction, limitations_of_traditional_programs).
teaches(introduction, introduction_to_machine_learning).
teaches(introduction, distinction_from_traditional_programming).
teaches(introduction, definition_of_machine_learning).
teaches(introduction, applications_of_machine_learning).
teaches(preliminaries, tensor_operations).
teaches(preliminaries, linear_algebra).
teaches(preliminaries, calculus_basics).
teaches(preliminaries, auto_diff).
teaches(preliminaries, probability_basics).
teaches(preliminaries, statistics_basics).
teaches(preliminaries, gradient_descent).
teaches(preliminaries, optimization).
teaches(preliminaries, backpropagation).
teaches(preliminaries, stochastic_models).
teaches(preliminaries, data_preprocessing).
teaches(preliminaries, matrix_multiplication).
teaches(preliminaries, broadcasting).
teaches(preliminaries, chain_rule).
teaches(preliminaries, loss_function_optimization).

% Facts defining the direct prerequisites for each topic and concept
hasPrerequisite(introduction, none).
hasPrerequisite(preliminaries, introduction).
hasPrerequisite(linear_neural_networks, preliminaries).
hasPrerequisite(multilayer_perceptrons, linear_neural_networks).
hasPrerequisite(builders_guide, multilayer_perceptrons).
hasPrerequisite(convolutional_neural_networks, builders_guide).
hasPrerequisite(modern_convolutional_neural_networks, convolutional_neural_networks).
hasPrerequisite(recurrent_neural_networks, builders_guide).
hasPrerequisite(modern_recurrent_neural_networks, recurrent_neural_networks).
hasPrerequisite(computer_vision, modern_convolutional_neural_networks).
hasPrerequisite(natural_language_processing, modern_recurrent_neural_networks).
hasPrerequisite(optimization_algorithms, builders_guide).
hasPrerequisite(computational_performance, optimization_algorithms).
hasPrerequisite(attention_mechanisms_and_transformers, builders_guide).
hasPrerequisite(limitations_of_traditional_programs, computer_programs_and_business_logic).
hasPrerequisite(introduction_to_machine_learning, limitations_of_traditional_programs).
hasPrerequisite(distinction_from_traditional_programming, introduction_to_machine_learning).
hasPrerequisite(definition_of_machine_learning, distinction_from_traditional_programming).
hasPrerequisite(applications_of_machine_learning, definition_of_machine_learning).
hasPrerequisite(tensor_operations, linear_algebra).
hasPrerequisite(matrix_multiplication, linear_algebra).
hasPrerequisite(data_preprocessing, tensor_operations).
hasPrerequisite(broadcasting, tensor_operations).
hasPrerequisite(chain_rule, calculus_basics).
hasPrerequisite(backpropagation, chain_rule).
hasPrerequisite(backpropagation, auto_diff).
hasPrerequisite(loss_function_optimization, backpropagation).
hasPrerequisite(loss_function_optimization, auto_diff).
hasPrerequisite(loss_function_optimization, optimization).
hasPrerequisite(statistics_basics, probability_basics).
hasPrerequisite(stochastic_models, statistics_basics).
hasPrerequisite(gradient_descent, calculus_basics).
hasPrerequisite(optimization, gradient_descent).

% Rule defining if a student can learn a specific topic or concept based on prerequisites
can_learn(Student, TopicOrConcept) :-
    hasPrerequisite(TopicOrConcept, Prerequisite),
    (Prerequisite == none ; learned(Student, Prerequisite)).

% Rule to check if a student has learned all prerequisites for a topic or concept
has_learned_all_prerequisites(Student, TopicOrConcept) :-
    hasPrerequisite(TopicOrConcept, Prerequisite),
    (Prerequisite == none ; (learned(Student, Prerequisite), has_learned_all_prerequisites(Student, Prerequisite))).

% Rule to suggest the next topic or concept to learn based on current knowledge
suggest_next_topic_or_concept(Student, NextTopicOrConcept) :-
    not(learned(Student, NextTopicOrConcept)),
    can_learn(Student, NextTopicOrConcept),
    has_learned_all_prerequisites(Student, NextTopicOrConcept).

% Rule to check if a student has learned all concepts in a section
has_learned_all_concepts(Student, Section) :-
    not((teaches(Section, Concept), not(learned(Student, Concept)))).

% Fact about students having learned certain topics and concepts
learned(john, computer_programs_and_business_logic).
learned(john, limitations_of_traditional_programs).
learned(jane, introduction_to_machine_learning).
learned(jane, distinction_from_traditional_programming).

% Example usage of the rule to suggest the next topic or concept to learn
suggest_next_topic_or_concept(john, NextTopicOrConcept).