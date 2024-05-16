% Define sections
section(introduction).
section(preliminaries).
section(linear_neural_networks).
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

% Define which section teaches which concept
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

% Define section prerequisites
hasSectionPrerequisite(introduction, none).
hasSectionPrerequisite(preliminaries, introduction).
hasSectionPrerequisite(linear_neural_networks, preliminaries).
hasSectionPrerequisite(multilayer_perceptrons, linear_neural_networks).
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

% Facts defining the direct prerequisites for each concept
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