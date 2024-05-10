% Define sections
section(introduction).
section(preliminaries).
section(linear_neural_networks_regression).
section(linear_neural_networks_classification).
section(multilayer_perceptrons).
section(builders_guide).
section(convolutional_neural_networks).
section(modern_convolutional_neural_networks).
section(recurrent_neural_networks).
section(modern_recurrent_neural_networks).
section(computer_vision).
section(natural_language_processing).
section(optimization_algorithms).
section(computational_performance).
section(attention_mechanisms_and_transformers).

% Define section prerequisites
hasSectionPrerequisite(introduction, none).
hasSectionPrerequisite(preliminaries, introduction).
hasSectionPrerequisite(linear_neural_networks_regression, preliminaries).
hasSectionPrerequisite(linear_neural_networks_classification, linear_neural_networks_regression).
hasSectionPrerequisite(multilayer_perceptrons, linear_neural_networks_classification).
hasSectionPrerequisite(builders_guide, multilayer_perceptrons).
hasSectionPrerequisite(convolutional_neural_networks, builders_guide).
hasSectionPrerequisite(modern_convolutional_neural_networks, convolutional_neural_networks).
hasSectionPrerequisite(recurrent_neural_networks, builders_guide).
hasSectionPrerequisite(modern_recurrent_neural_networks, recurrent_neural_networks).
hasSectionPrerequisite(computer_vision, modern_convolutional_neural_networks).
hasSectionPrerequisite(natural_language_processing, modern_recurrent_neural_networks).
hasSectionPrerequisite(optimization_algorithms, builders_guide).
hasSectionPrerequisite(computational_performance, optimization_algorithms).
hasSectionPrerequisite(attention_mechanisms_and_transformers, builders_guide).

% --------- Introduction section ---------
teachesConcept(introduction, computer_programs_and_business_logic).
teachesConcept(introduction, limitations_of_traditional_programs).
teachesConcept(introduction, introduction_to_machine_learning).
teachesConcept(introduction, distinction_from_traditional_programming).
teachesConcept(introduction, definition_of_machine_learning).
teachesConcept(introduction, applications_of_machine_learning).
teachesConcept(preliminaries, tensor_operations).
teachesConcept(preliminaries, linear_algebra).
teachesConcept(preliminaries, calculus_basics).
teachesConcept(preliminaries, auto_diff).

hasPrerequisite(limitations_of_traditional_programs, computer_programs_and_business_logic).
hasPrerequisite(introduction_to_machine_learning, limitations_of_traditional_programs).
hasPrerequisite(distinction_from_traditional_programming, introduction_to_machine_learning).
hasPrerequisite(definition_of_machine_learning, distinction_from_traditional_programming).
hasPrerequisite(applications_of_machine_learning, definition_of_machine_learning).

% --------- Preliminaries section ---------
teachesConcept(preliminaries, probability_basics).
teachesConcept(preliminaries, statistics_basics).
teachesConcept(preliminaries, gradient_descent).
teachesConcept(preliminaries, optimization).
teachesConcept(preliminaries, backpropagation).
teachesConcept(preliminaries, stochastic_models).
teachesConcept(preliminaries, data_preprocessing).
teachesConcept(preliminaries, matrix_multiplication).
teachesConcept(preliminaries, broadcasting).
teachesConcept(preliminaries, chain_rule).
teachesConcept(preliminaries, loss_function_optimization).

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

% --------- Linear neural networks regression section ---------
teachesConcept(linear_neural_networks_regression, neural_networks_for_regression).
teachesConcept(linear_neural_networks_regression, linear_regression).
teachesConcept(linear_neural_networks_regression, basics_of_linear_regression).
teachesConcept(linear_neural_networks_regression, vectorization_for_speed).
teachesConcept(linear_neural_networks_regression, normal_distribution_and_squared_loss).
teachesConcept(linear_neural_networks_regression, linear_regression_as_a_neural_network).
teachesConcept(linear_neural_networks_regression, object_oriented_design).
teachesConcept(linear_neural_networks_regression, utilities).
teachesConcept(linear_neural_networks_regression, models).
teachesConcept(linear_neural_networks_regression, data).
teachesConcept(linear_neural_networks_regression, training).
teachesConcept(linear_neural_networks_regression, synthetic_data_generation).
teachesConcept(linear_neural_networks_regression, dataset_for_regression).
teachesConcept(linear_neural_networks_regression, implementation_from_scratch).
teachesConcept(linear_neural_networks_regression, training_error_and_generalization_error).
teachesConcept(linear_neural_networks_regression, model_complexity).
teachesConcept(linear_neural_networks_regression, underfitting_and_overfitting).
teachesConcept(linear_neural_networks_regression, polynomial_curve_fitting).
teachesConcept(linear_neural_networks_regression, concise_implementation).
teachesConcept(linear_neural_networks_regression, high_level_apis).
teachesConcept(linear_neural_networks_regression, frameworks_for_deep_learning).
teachesConcept(linear_neural_networks_regression, generalization_in_machine_learning).
teachesConcept(linear_neural_networks_regression, training_error_vs_generalization_error).
teachesConcept(linear_neural_networks_regression, model_selection).
teachesConcept(linear_neural_networks_regression, cross_validation).
teachesConcept(linear_neural_networks_regression, weight_decay).
teachesConcept(linear_neural_networks_regression, regularization_techniques).
teachesConcept(linear_neural_networks_regression, l2_norm_penalty).
teachesConcept(linear_neural_networks_regression, high_dimensional_linear_regression).

hasPrerequisite(basics_of_linear_regression, linear_regression).
hasPrerequisite(vectorization_for_speed, linear_regression).
hasPrerequisite(normal_distribution_and_squared_loss, linear_regression).
hasPrerequisite(linear_regression_as_a_neural_network, linear_regression).
hasPrerequisite(utilities, object_oriented_design).
hasPrerequisite(models, object_oriented_design).
hasPrerequisite(data, object_oriented_design).
hasPrerequisite(training, object_oriented_design).
hasPrerequisite(dataset_for_regression, synthetic_data_generation).
hasPrerequisite(implementation_from_scratch, linear_regression).
hasPrerequisite(training_error_and_generalization_error, implementation_from_scratch).
hasPrerequisite(model_complexity, implementation_from_scratch).
hasPrerequisite(underfitting_and_overfitting, implementation_from_scratch).
hasPrerequisite(polynomial_curve_fitting, implementation_from_scratch).
hasPrerequisite(concise_implementation, high_level_apis).
hasPrerequisite(high_level_apis, frameworks_for_deep_learning).
hasPrerequisite(training_error_vs_generalization_error, generalization_in_machine_learning).
hasPrerequisite(model_selection, generalization_in_machine_learning).
hasPrerequisite(cross_validation, model_selection).
hasPrerequisite(weight_decay, regularization_techniques).
hasPrerequisite(l2_norm_penalty, weight_decay).
hasPrerequisite(high_dimensional_linear_regression, weight_decay).


% =================== Helper functions ===================


% A student has learned a section if they have learned all concepts in that section
hasLearnedSection(Student, Section) :-
    section(Section),
    not((
        teachesConcept(Section, Concept),
        not(hasLearnedConcept(Student, Concept))
    )).

% A student can learn a section if they have learned all dependent sections
canLearnSection(Student, Section) :-
    section(Section),
    (
        % Check if the section has a 'none' prerequisite directly
        hasSectionPrerequisite(Section, none)
        ;
        % Check if all prerequisites are met
        not((
            hasSectionPrerequisite(Section, PrerequisiteSection),
            PrerequisiteSection \= none,
            not(hasLearnedSection(Student, PrerequisiteSection))
        ))
    ).

% A student can learn a concept if they can learn the section that teaches it, and if they have learned all prerequisites
canLearnConcept(Student, Concept) :-
    teachesConcept(Section, Concept),
    canLearnSection(Student, Section),
    not(hasLearnedConcept(Student, Concept)),
    not((
        hasPrerequisite(Concept, PrerequisiteConcept),
        not(hasLearnedConcept(Student, PrerequisiteConcept))
    )).

% A next step should be a learnable concept that will bring the student closer to their goal
% If the goal is a section, the student needs to learn all prerequisite sections, and then all of the concepts in the section itself
% If the goal is a concept, the student needs to learn all prerequisite sections, and then learn the prerequisite concepts in the section
% Determine if all direct prerequisites of a concept have been learned by the student

% Recursively find all prerequisites leading up to a concept.
prerequisitePath(GoalConcept, Concept) :-
    hasPrerequisite(GoalConcept, Concept).
prerequisitePath(GoalConcept, Concept) :-
    hasPrerequisite(GoalConcept, Intermediate),
    prerequisitePath(Intermediate, Concept).

% Check if all prerequisites for a concept have been learned by the student.
allPrerequisitesLearned(Student, Concept) :-
    not((hasPrerequisite(Concept, Prerequisite),
         not(hasLearnedConcept(Student, Prerequisite)))).

prerequisitesMet(Student, GoalConcept, Concept) :-
    prerequisitePath(GoalConcept, Concept),
    not(hasLearnedConcept(Student, Concept)),
    allPrerequisitesLearned(Student, Concept).

% Identifies the next learnable concept for the student based on the direct and indirect prerequisites
% required for reaching the goal concept, ensuring uniqueness in suggestions.
nextStep(Student, NextConcept) :-
    hasGoal(Student, GoalConcept),
    setof(Concept, prerequisitesMet(Student, GoalConcept, Concept), UniqueConcepts),
    member(NextConcept, UniqueConcepts).

% Optionally, consider the prerequisites of the section if the goal concept's prerequisites are satisfied
nextStep(Student, NextSection) :-
    hasGoal(Student, GoalConcept),
    teachesConcept(Section, GoalConcept),
    setof(SectionConcept,
          sectionPrerequisitesMet(Student, Section, GoalConcept, SectionConcept),
          UniqueSections),
    member(NextSection, UniqueSections).

sectionPrerequisitesMet(Student, Section, GoalConcept, PrerequisiteSection) :-
    hasSectionPrerequisite(Section, PrerequisiteSection),
    PrerequisiteSection \= none,
    not(hasLearnedSection(Student, PrerequisiteSection)),
    teachesConcept(PrerequisiteSection, NecessaryConcept),
    prerequisitePath(GoalConcept, NecessaryConcept),
    not(hasLearnedConcept(Student, NecessaryConcept)),
    PrerequisiteSection = Section.

% =================== Example data ===================

student(john).
hasGoal(john, loss_function_optimization).
hasLearnedConcept(john, computer_programs_and_business_logic).
hasLearnedConcept(john, limitations_of_traditional_programs).
hasLearnedConcept(john, introduction_to_machine_learning).
hasLearnedConcept(john, distinction_from_traditional_programming).
hasLearnedConcept(john, definition_of_machine_learning).
hasLearnedConcept(john, applications_of_machine_learning).
hasLearnedConcept(john, calculus_basics).