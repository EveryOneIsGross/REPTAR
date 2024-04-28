# REPTAR
Recursive / Enriching / Pterodactyl / Tree / Augmented / Retrieval

![image](https://github.com/EveryOneIsGross/REPTAR/assets/23621140/53afdc05-b4c9-4941-b9f7-3b92b2d25575)

REPTAR is a system that uses a recursive summarization approach to generate thoughtful summaries of text data. The system first preprocesses the text data, constructs a hierarchical graph, and performs cluster summarization. It then enriches the cluster summaries by appending similar text chunks, iteratively combines and summarizes the enriched cluster summaries, and generates a final thoughtful summary. The system also updates the corpus and embeddings based on the generated summaries and user queries. The system includes a conversation loop where the user can interact with the system using a conversation agent.


```mermaid
 graph TD
    A[Text Preprocessing] --> B[Word Embedding]
    B --> C[Clustering and Graph Construction]
    C --> D[Cluster Summarization]
    D --> E[Enrich Cluster Summaries]
    E --> F[Convolutional Summarization]
    F --> G[Flatten Convolutional Summaries]
    G --> H[Enrich Flattened Convolutional Summaries]
    H --> I[Final Thoughtful Summary]
    I --> J[Save Summaries to JSON]
    J --> K[Update Corpus and Embeddings]
    K --> L[Conversation Loop]

    A --> M[Tokenization, Lowercasing, Removing Non-Alphabetic Characters]
    M --> N[Split Text into Chunks]
    N --> O[Associate Chunks with Source File Paths]

    B --> P[Train Word2Vec Model]
    P --> Q[Generate Word Embeddings]

    C --> R[Calculate Chunk Embeddings]
    R --> S[Perform Hierarchical Clustering]
    S --> T[Assign Cluster Labels]
    T --> U[Construct Directed Graph]

    D --> V[Aggregate Text of Chunks within Each Cluster]
    V --> W[Generate Cluster Summaries]

    E --> X[Append Similar Text Chunks to Cluster Summaries]

    F --> Y[Iteratively Combine and Summarize Enriched Cluster Summaries]
    Y --> Z[Store Intermediate Convolutional Summaries at Each Depth Level]

    G --> AA[Flatten Convolutional Summaries List]

    H --> AB[Append Similar Text Chunks to Flattened Convolutional Summaries]

    I --> AC[Generate Reflective and Comprehensive Summary]

    K --> AD[Chunk Non-Enhanced Summaries into Sentences]
    AD --> AE[Add Summary Sentences to Original Corpus]
    AE --> AF[Retrain Word2Vec Model with Updated Corpus]
    AF --> AG[Calculate Updated Chunk Embeddings]

    L --> AH[User Enters Query]
    AH --> AI{Query Type}
    AI -- Exit/Quit --> AJ[End Conversation Loop]
    AI -- Save --> AK[Save Updated Embeddings, Model, and Settings]
    AI -- Other --> AL[Generate Response using Conversation Agent]
    AL --> AH
```


![image](https://github.com/EveryOneIsGross/REPTAR/assets/23621140/736d55a3-5a49-4a42-9dc9-10d058716ec1)



