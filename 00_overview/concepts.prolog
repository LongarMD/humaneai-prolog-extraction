% Facts defining the direct prerequisites for each topic
prerequisite(introduction, none).
prerequisite(preliminaries, introduction).
prerequisite(linear_neural_networks, preliminaries).
prerequisite(multilayer_perceptrons, linear_neural_networks).
prerequisite(builders_guide, multilayer_perceptrons).
prerequisite(convolutional_neural_networks, builders_guide).
prerequisite(modern_convolutional_neural_networks, convolutional_neural_networks).
prerequisite(recurrent_neural_networks, builders_guide).
prerequisite(modern_recurrent_neural_networks, recurrent_neural_networks).
prerequisite(computer_vision, modern_convolutional_neural_networks).
prerequisite(natural_language_processing, modern_recurrent_neural_networks).
prerequisite(optimization_algorithms, builders_guide).
prerequisite(computational_performance, optimization_algorithms).
prerequisite(attention_mechanisms_and_transformers, builders_guide).

% Rules defining if a student can learn a topic based on prerequisites
can_learn(Student, Topic) :-
    prerequisite(Topic, Prerequisite),
    (Prerequisite == none ; learned(Student, Prerequisite)).

% Rule to check if a student has learned all prerequisites for a topic
has_learned_all_prerequisites(Student, Topic) :-
    prerequisite(Topic, Prerequisite),
    (Prerequisite == none ; (learned(Student, Prerequisite), has_learned_all_prerequisites(Student, Prerequisite))).

% Rule to suggest the next topic to learn based on current knowledge
suggest_next_topic(Student, NextTopic) :-
    not(learned(Student, NextTopic)),
    can_learn(Student, NextTopic),
    has_learned_all_prerequisites(Student, NextTopic).
    
learned(john, intro).
