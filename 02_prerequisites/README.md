### Prompts

- `From the following chapter "Preliminaries" of the Dive into Deep Learning book, extract the core concepts a student needs to master before delving deeper into the book. If concepts are connected, e.g. you need to understand 1st order derivatives before you can understand partial derivatives, be sure to make note of that.`
- `Now, generate a new list of very specific concepts and construct a graph of a student's learning flow using the graphviz notation. Where possible, use OpenCyc concepts and relations`
- Enhance the concepts and relationships with this set of opencyc relationships:
  ```
  isPrerequisiteFor:
  This relation indicates that understanding one concept or topic is necessary before comprehending another.
  Example: isPrerequisiteFor(WordEmbeddings, NeuralNetworks) implies that understanding word embeddings is necessary before comprehending neural networks.
  isApplicationOf:
  This relation signifies that a particular concept or technique is applied in a certain context or field.
  Example: isApplicationOf(AttentionMechanism, NaturalLanguageProcessing) implies that attention mechanisms are applied in natural language processing tasks.
  isDiscipline:
  This relation categorizes concepts or topics into specific disciplines or fields of study.
  Example: isDiscipline(NeuralNetworks, MachineLearning) indicates that neural networks belong to the field of machine learning.
  isEvolutionOf:
  This relation captures the evolutionary progression of concepts or technologies.
  Example: isEvolutionOf(NeuralNetworks, Perceptrons) suggests that neural networks have evolved from the concept of perceptrons.
  isSubtopicOf:
  This relation indicates that one concept or topic is a subtopic of another, providing a hierarchical structure.
  Example: isSubtopicOf(TransformerNetworks, NeuralNetworks) implies that transformer networks are a subtopic of neural networks.
  isMethodOf:
  This relation signifies that a particular method is employed in achieving a certain task or objective.
  Example: isMethodOf(Backpropagation, TrainingNeuralNetworks) suggests that backpropagation is a method used for training neural networks.
  isComponentOf:
  This relation indicates that one entity is a fundamental part or component of another.
  Example: isComponentOf(Neuron, NeuralNetwork) implies that neurons are components of neural networks.
  isFormulationOf:
  This relation signifies that one concept or theory is a formalized expression of another.
  Example: isFormulationOf(WordEmbeddings, DistributionalSemantics) indicates that word embeddings are a formalized expression of distributional semantics.
  isEnhancementOf:
  This relation captures the improvement or enhancement of one concept or technique over another.
  Example: isEnhancementOf(GPT-3, GPT-2) suggests that GPT-3 is an enhancement over GPT-2.
  isIncorporationOf:
  This relation indicates the integration or inclusion of one concept or technique into another.
  Example: isIncorporationOf(AttentionMechanism, RecurrentNeuralNetworks) implies that attention mechanisms have been incorporated into recurrent neural networks.
  ```
