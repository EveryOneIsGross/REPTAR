
# Recursive Enriching Pterodactyl Tree Augmented Retrieval (REPTAR)   is a system that uses a recursive summarization approach to generate thoughtful summaries of text data. The system first preprocesses the text data, constructs a hierarchical graph, and performs cluster summarization. It then enriches the cluster summaries by appending similar text chunks, iteratively combines and summarizes the enriched cluster summaries, and generates a final thoughtful summary. The system also updates the corpus and embeddings based on the generated summaries and user queries. The system includes a conversation loop where the user can interact with the system using a conversation agent.

'''graph TD
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
    AL --> AH'''

'''
Noted bugs:
- Final summary not quite refined right, check inputs.
- prompts need werk
- tracks " , " as a source when reading outputs. Dunno where this is coming from
'''

import gensim
from gensim.models import Word2Vec
from scipy.spatial.distance import pdist, squareform, cosine
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter
import networkx as nx
import pickle
import smart_open
import numpy as np
from openai import OpenAI
import spacy
import json
import os
import datetime
import re



chat_history = []

file_paths = ['socialnetwork.txt', 'crow.txt']

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.add_pipe('sentencizer')

TEH_CHUNK_SIZE = 512


def read_and_preprocess(file_paths, CHUNK_SIZE=TEH_CHUNK_SIZE):
    for file_path in file_paths:
        with smart_open.smart_open(file_path, encoding="utf-8") as f:
            text = f.read()
            doc = nlp(text)
            chunk = []
            for token in doc:
                if token.is_alpha:
                    chunk.append(token.text.lower())
                    if len(chunk) >= CHUNK_SIZE:
                        yield (chunk[:CHUNK_SIZE], file_path)
                        chunk = chunk[CHUNK_SIZE:]
            if chunk:
                yield (chunk, file_path)

def get_chunk_vector(model, chunk):
    word_vectors = [model.wv[word] for word in chunk if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

def find_matching_nodes(graph, query_vector, top_k=1):
    distances = []
    for node in graph:
        if 'text' in graph.nodes[node]:
            node_vector = get_chunk_vector(model, gensim.utils.simple_preprocess(graph.nodes[node]['text']))
            distance = cosine(query_vector, node_vector)
            source_file = graph.nodes[node].get('source_file', 'Unknown')
            distances.append((node, distance, source_file))
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]

def aggregate_cluster_content(graph, cluster_id):
    texts = []
    for node in graph.nodes:
        if graph.nodes[node].get('cluster') == cluster_id:
            texts.append(graph.nodes[node].get('text', ''))
    aggregated_text = " ".join(texts)
    return aggregated_text

def thoughtful_summary_agent(text, user_query, max_tokens=32000):
    global chat_history

    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )

    history_prompt = "\n".join([f"{item['role']}: {item['content']}" for item in chat_history])

    if not history_prompt:
        history_prompt = "You just awoke, answer thoughtfully when discussing your inner thoughts with others. Answer questions with intense thoughtfulness."

    prompt = f"You are a thoughtful chatbot with that self reflects on their output.\n\nReflecting on past thoughts:\n\n{history_prompt}\n\nIntegrate current thoughts:\n\n{text}\n"

    response = client.chat.completions.create(
        model="mistral:instruct",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"\n\nReflect on thoughts and answer the following:\n\n{user_query}\n\n"}
        ],
        max_tokens=32000,
        temperature=0.2,
    )

    summary = response.choices[0].message.content

    chat_history.append({"role": "system", "content": prompt})
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": summary})

    chat_history = chat_history[-4:]

    return summary

def chunk_summary_agent(chunk_text, user_query, max_tokens=256):
    global chat_history
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )

    system_prompt = f"You are a thoughtful systems oriented summary agent and can see the potential connections between everything. Assume you have everything you need to reply."

    prompt = f"Here is your stream of consciousness '{chunk_text}', organise your thoughts into a single concise factoid.\n\n"

    response = client.chat.completions.create(
        model="tinydolphin:latest",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt + "\n\n"},
        ],
        max_tokens=1024,
        temperature=0.2
    )

    summary = response.choices[0].message.content

    cleaned_summary = clean_text(summary)
    print(f"Node Summary: {cleaned_summary}")
    chat_history.append({"role": "assistant", "content": cleaned_summary})

    return summary

def clean_text(text):
    phrases_to_remove = ["I'm sorry "]
    for phrase in phrases_to_remove:
        text = text.replace(phrase, "")
    text = ' '.join(word for word in text.split() if not word.isdigit() and not word.replace('.', '', 1).isdigit())
    text = text.replace('\n', ' ')
    text = text.replace('\n\n', ' ')
    text = ' '.join(text.split())

    text = re.sub(r'[^a-zA-Z0-9.,\s]', '', text)
    text = text.strip()

    return text

def get_cluster_information_with_summary(graph, cluster_id, query_vector, model, top_k=5, user_query="", max_chunk_summaries=3):
    aggregated_content = aggregate_cluster_content(graph, cluster_id)

    cluster_summary = thoughtful_summary_agent(aggregated_content, user_query, max_tokens=TEH_CHUNK_SIZE)

    cluster_documents = []

    for node in graph.nodes:
        if graph.nodes[node].get('cluster') == cluster_id:
            node_vector = get_chunk_vector(model, gensim.utils.simple_preprocess(graph.nodes[node].get('text', '')))
            similarity = 1 - cosine(query_vector, node_vector)
            source_file = graph.nodes[node].get('source_file', 'No source file available')
            cluster_documents.append((graph.nodes[node].get('text', 'No text available'), similarity, source_file))

    sorted_documents = sorted(cluster_documents, key=lambda x: x[1], reverse=True)
    top_documents_with_source = [(doc[0], doc[2]) for doc in sorted_documents[:top_k]]

    chunk_summaries = []
    for doc_text, _ in top_documents_with_source[:max_chunk_summaries]:
        chunk_summary = chunk_summary_agent(doc_text, user_query, max_tokens=TEH_CHUNK_SIZE*2)
        chunk_summaries.append(chunk_summary)

    return cluster_summary, top_documents_with_source, chunk_summaries

def chat_with_embedding_from_pickle(query, model, top_k=5, pickle_path="hierarchical_graph.pkl", user_query=""):
    with open(pickle_path, "rb") as f:
        graph = pickle.load(f)
    query_vector = get_chunk_vector(model, gensim.utils.simple_preprocess(query))

    matching_nodes = find_matching_nodes(graph, query_vector, top_k=3)

    if matching_nodes:
        top_match = matching_nodes[0][0]
        cluster_id = graph.nodes[top_match].get('cluster')

        if cluster_id:
            cluster_summary, top_documents_with_source, chunk_summaries = get_cluster_information_with_summary(graph, cluster_id, query_vector, model, top_k, user_query)
            cluster_summary = clean_text(cluster_summary)
            response_text = f"{cluster_summary}\n\n" + "\n".join([f"- {doc_text} (Source: {source_file})" for doc_text, source_file in top_documents_with_source])

            return response_text, cluster_summary, chunk_summaries
        else:
            return "Found a relevant section, but couldn't locate its broader context within the cluster.", None, None
    else:
        return "Unable to find relevant information for the query.", None, None

def serialize_kg_for_chat(kg_json):
    entities_summary = '; '.join([f"{entity}: {details['type']} (Weight: {details.get('weight', 0.0):.4f})" for entity, details in kg_json['entities'].items()])
    relationships_summary = '; '.join([f"{rel['subject']} {rel['verb']} {rel['object']} (Weight: {rel.get('weight', 0.0):.4f})" for rel in kg_json['relationships']])
    return f"KG Entities: {entities_summary}. KG Relationships: {relationships_summary}."

def handle_query(user_query, model, graph, top_k=5):
    global chat_history

    query_vector = get_chunk_vector(model, gensim.utils.simple_preprocess(user_query))

    matching_nodes = find_matching_nodes(graph, query_vector, top_k=3)
    cluster_ids = set(graph.nodes[node[0]].get('cluster') for node in matching_nodes)
    filtered_nodes = [node for node in graph.nodes if graph.nodes[node].get('cluster') in cluster_ids]

    expanded_query = user_query
    for node in filtered_nodes:
        chunk_text = graph.nodes[node].get('text', '')
        expanded_query += ' ' + ' '.join(gensim.utils.simple_preprocess(chunk_text))

    response_text, cluster_summary, chunk_summaries = chat_with_embedding_from_pickle(expanded_query, model, top_k, "hierarchical_graph.pkl", user_query)

    if not isinstance(chunk_summaries, list):
        chunk_summaries = []

    if response_text:
        formatted_response = f"{response_text}\n\n"
        for i, summary in enumerate(chunk_summaries, start=1):
            formatted_response += f"{i}. {summary}\n"

        combined_text = response_text

        kg_generator = EnhancedKnowledgeGraphGenerator()
        kg_json = kg_generator.process_text(combined_text)

        kg_summary = serialize_kg_for_chat(kg_json)

        cleaned_kg_summary = clean_text(kg_summary)

        chat_history.append({"role": "system", "content": cleaned_kg_summary})
        chat_history = chat_history[-5:]

    return formatted_response

class EnhancedKnowledgeGraphGenerator:
    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)

    def process_text(self, graph_text):
        doc = self.nlp(graph_text)
        word_counts = Counter(token.text.lower() for token in doc if not token.is_stop and not token.is_punct)
        total_words = sum(word_counts.values())

        spacy_entities = self.extract_entities(doc, word_counts, total_words)
        custom_entities = self.extract_custom_entities(doc, spacy_entities, word_counts, total_words)
        all_entities = {**spacy_entities, **custom_entities}
        relationships = self.extract_relationships(doc, word_counts, total_words)

        all_entities, relationships = self.infer_missing_relationships(doc, all_entities, relationships)

        return self.create_kg_json(all_entities, relationships)

    def extract_entities(self, doc, word_counts, total_words):
        entities = {}
        for ent in doc.ents:
            weight = word_counts.get(ent.text.lower(), 0) / total_words if total_words > 0 else 0
            entities[ent.text] = {"type": ent.label_, "weight": weight}
        return entities

    def extract_custom_entities(self, doc, existing_entities, word_counts, total_words):
        entities = {}
        for token in doc:
            if token.pos_ == "PROPN" and token.text not in existing_entities:
                weight = word_counts.get(token.text.lower(), 0) / total_words if total_words > 0 else 0
                entities[token.text] = {"type": "ENTITY", "weight": weight}
        return entities

    def extract_relationships(self, doc, word_counts, total_words):
        relationships = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB":
                    subjects = [child for child in token.children if child.dep_ == "nsubj"]
                    objects = [child for child in token.children if child.dep_ in ("dobj", "attr", "prep")]
                    for subj in subjects:
                        for obj in objects:
                            weight = (word_counts.get(subj.text.lower(), 0) + word_counts.get(obj.text.lower(), 0)) / (2 * total_words) if total_words > 0 else 0
                            relationships.append({"subject": subj.text, "verb": token.lemma_, "object": obj.text, "weight": weight})
        return relationships

    def infer_missing_relationships(self, doc, entities, relationships):
        inferred_relationships = []

        for sent in doc.sents:
            sentence_entities = {ent.text for ent in sent.ents if ent.text in entities}
            if len(sentence_entities) > 1:
                for entity in sentence_entities:
                    for other_entity in sentence_entities:
                        if entity != other_entity:
                            inferred_relationship = {"subject": entity, "verb": "association", "object": other_entity}
                            if inferred_relationship not in relationships:
                                inferred_relationships.append(inferred_relationship)

        relationships.extend(inferred_relationships)
        return entities, relationships

    def create_kg_json(self, entities, relationships):
        kg_json = {"entities": entities, "relationships": relationships}
        return kg_json


chunks_with_paths = list(read_and_preprocess(file_paths))
corpus = [chunk for chunk, _ in chunks_with_paths]
source_file_paths = [source_file_path for _, source_file_path in chunks_with_paths]

model = Word2Vec(sentences=corpus, vector_size=256, window=10, min_count=1, workers=4)

chunk_vectors = [get_chunk_vector(model, chunk) for chunk in corpus]
distance_matrix = pdist(chunk_vectors, 'euclidean')
Z = linkage(distance_matrix, 'ward')
clusters = fcluster(Z, t=16, criterion='maxclust')
H = nx.DiGraph()

for i, cluster_id in enumerate(clusters):
    parent_node = f"Cluster_{cluster_id}"
    child_node = f"Doc_{i}"
    source_file_path = source_file_paths[i]
    source_file_name = os.path.basename(source_file_path)

    H.add_node(parent_node)
    H.add_node(child_node, text=' '.join(corpus[i]), cluster=cluster_id, source_file=source_file_name)
    H.add_edge(parent_node, child_node)

with open("hierarchical_graph.pkl", "wb") as f:
    pickle.dump(H, f)

def cluster_summarization(graph, model, source_file_paths):
    summaries = []
    for cluster_id in set(graph.nodes[node].get('cluster') for node in graph.nodes):
        cluster_nodes = [node for node in graph.nodes if graph.nodes[node].get('cluster') == cluster_id]
        cluster_text = ' '.join([graph.nodes[node].get('text', '') for node in cluster_nodes])
        cluster_source_files = [graph.nodes[node].get('source_file', '') for node in cluster_nodes]
        summary = chunk_summary_agent(cluster_text, '', max_tokens=TEH_CHUNK_SIZE)
        summaries.append({
            'text': summary,
            'source_files': cluster_source_files,
            'depth': 0
        })
    return summaries

def recursive_cluster_summarization(graph, model, source_file_paths, depth=0):
    summaries = []
    for cluster_id in set(graph.nodes[node].get('cluster') for node in graph.nodes):
        cluster_nodes = [node for node in graph.nodes if graph.nodes[node].get('cluster') == cluster_id]
        if len(cluster_nodes) == 1:
            node = cluster_nodes[0]
            summaries.append({
                'text': graph.nodes[node].get('text', ''),
                'source_file': graph.nodes[node].get('source_file', ''),
                'depth': depth
            })
        else:
            cluster_summaries = []
            for node in cluster_nodes:
                chunk_text = graph.nodes[node].get('text', '')
                source_file = graph.nodes[node].get('source_file', '')
                summary = chunk_summary_agent(chunk_text, '', max_tokens=TEH_CHUNK_SIZE)
                cluster_summaries.append({
                    'text': summary,
                    'source_file': source_file,
                    'depth': depth
                })
            summaries.extend(recursive_cluster_summarization(graph.subgraph(cluster_nodes), model, source_file_paths, depth+1))
    return summaries

def convolutional_summarization(summaries, depth):
    convolutional_summaries = []
    current_summaries = summaries

    while len(current_summaries) > 1:
        new_summaries = []
        for i in range(0, len(current_summaries), 2):
            if i + 1 < len(current_summaries):
                combined_text = current_summaries[i]['text'] + ' ' + current_summaries[i+1]['text']
                summary = chunk_summary_agent(combined_text, '', max_tokens=TEH_CHUNK_SIZE)
                new_summaries.append({
                    'text': summary,
                    'source_files': current_summaries[i]['source_files'] + current_summaries[i+1]['source_files'],
                    'depth': depth
                })
            else:
                new_summaries.append(current_summaries[i])
        current_summaries = new_summaries
        depth += 1
        convolutional_summaries.append(current_summaries)

    return convolutional_summaries, depth

def thoughtful_summarization(summary, depth):
    thoughtful_summary = thoughtful_summary_agent(summary['text'], '', max_tokens=TEH_CHUNK_SIZE)
    kg_generator = EnhancedKnowledgeGraphGenerator()
    kg_json = kg_generator.process_text(thoughtful_summary)
    return {
        'text': thoughtful_summary,
        'source_files': summary['source_files'],
        'depth': depth,
        'kg_json': kg_json
    }

def save_to_json(data, filename):
    def process_source_files(summary):
        source_files_count = Counter(summary['source_files'])
        return [{'file': source_file, 'mentions': count} for source_file, count in source_files_count.items()]

    output_data = {
        'cluster_summaries': [
            {
                'text': summary['text'],
                'source_files': process_source_files(summary),
                'depth': summary['depth'],
                'kg_json': summary.get('kg_json', None)
            }
            for summary in data['cluster_summaries']
        ],
        'enriched_cluster_summaries': [
            {
                'text': summary['text'],
                'enriched_text': summary['enriched_text'],
                'source_files': process_source_files(summary),
                'depth': summary['depth'],
                'kg_json': summary['kg_json']
            }
            for summary in data['enriched_cluster_summaries']
        ],
        'convolutional_summaries': [
            [
                {
                    'text': summary['text'],
                    'source_files': process_source_files(summary),
                    'depth': summary['depth'],
                    'kg_json': summary.get('kg_json', None)
                }
                for summary in depth_summaries
            ]
            for depth_summaries in data['convolutional_summaries']
        ],
        'enriched_convolutional_summaries': [
            {
                'text': summary['text'],
                'enriched_text': summary['enriched_text'],
                'source_files': process_source_files(summary),
                'depth': summary['depth'],
                'kg_json': summary['kg_json']
            }
            for summary in data['enriched_convolutional_summaries']
        ],
        'final_summary': {
            'text': data['final_summary']['text'],
            'source_files': process_source_files(data['final_summary']),
            'depth': data['final_summary']['depth'],
            'kg_json': data['final_summary'].get('kg_json', None)
        }
    }

    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=4)

def enrich_summary(summary, chunk_embeddings, corpus, top_k=3):
    summary_embedding = get_chunk_vector(model, gensim.utils.simple_preprocess(summary['text']))
    similarities = cosine_similarity([summary_embedding], chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    enriched_text = summary['text']
    for index in top_indices:
        enriched_text += ' ' + ' '.join(corpus[index])
    
    kg_generator = EnhancedKnowledgeGraphGenerator()
    kg_json = kg_generator.process_text(enriched_text)
    
    return {
        'text': summary['text'],
        'enriched_text': enriched_text,
        'source_files': summary['source_files'],
        'depth': summary['depth'],
        'kg_json': kg_json
    }

def conversation_agent(user_query, updated_model, updated_corpus, top_k=5, max_tokens=150):
    # Assuming the existence of the following functions and variables:
    # OpenAI, get_chunk_vector, cosine_similarity, gensim, nlp
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )

    # Embed the user query
    query_embedding = get_chunk_vector(updated_model, gensim.utils.simple_preprocess(user_query))

    # Find the top-k most similar chunks from the updated corpus
    updated_corpus_embeddings = [get_chunk_vector(updated_model, gensim.utils.simple_preprocess(" ".join(chunk))) for chunk in updated_corpus]
    similarities = cosine_similarity([query_embedding], updated_corpus_embeddings)[0]
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    top_k_chunks = [updated_corpus[i] for i in top_k_indices]

    # Process the top-k chunks into properly formatted sentences
    context_sentences = []
    for chunk in top_k_chunks:
        # Ensure chunk is a string before processing
        chunk_text = " ".join(chunk) if isinstance(chunk, list) else chunk
        doc = nlp(chunk_text)
        for sent in doc.sents:
            context_sentences.append(sent.text.capitalize())

    # Create the conversation style system prompt
    system_prompt = "You are an AI assistant with access to relevant information from various sources. Your role is to provide helpful and insightful responses based on the given context. If the context doesn't contain enough information to answer the query, kindly let the user know that you don't have sufficient information to provide a complete answer."

    # Construct the prompt with the context sentences
    context = "\n".join([f"Context {i+1}: {sentence}" for i, sentence in enumerate(context_sentences)])
    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser: {user_query}\n"
    print(f"\nPrompt: {prompt}\n")
    # Generate the response using the language model
    response = client.chat.completions.create(
        model="mistral:instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )

    # Extract the assistant's response
    assistant_response = response.choices[0].message.content.strip()

    return assistant_response

def save_embeddings(embeddings, custom_name):
    filename = f"{custom_name}_embeddings.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

def save_model(model, custom_name):
    filename = f"{custom_name}_model.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def save_settings(settings, custom_name):
    filename = f"{custom_name}_settings.json"
    with open(filename, 'w') as f:
        json.dump(settings, f, indent=4)

def main():
    if input("Do you want to load previous embeddings and model for chat? (y/n): ").lower() == 'y':
        custom_name = input("Enter the name of the saved data (without file extension): ")
        embeddings_file = f"{custom_name}_embeddings.pkl"
        model_file = f"{custom_name}_model.pkl"
        settings_file = f"{custom_name}_settings.json"
        
        if os.path.exists(embeddings_file) and os.path.exists(model_file) and os.path.exists(settings_file):
            with open(embeddings_file, 'rb') as f:
                chunk_embeddings = pickle.load(f)
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            
            corpus = settings['corpus']
            source_file_paths = settings.get('source_file_paths', [])  # Use an empty list if not available
            
            # Skip processing steps and go directly to the conversation loop
            print("Conversation Loop:")
            while True:
                user_query = input("User: ")
                if user_query.lower() in ['exit', 'quit']:
                    break
                elif user_query.lower().startswith('save '):
                    custom_name = user_query.split(' ', 1)[1]
                    save_embeddings(chunk_embeddings, custom_name)
                    save_model(model, custom_name)
                    save_settings({
                        'corpus': corpus,
                        'source_file_paths': source_file_paths,
                        'vector_size': 256,
                        'window': 10,
                        'min_count': 1,
                        'workers': 4
                    }, custom_name)
                    print(f"Embeddings, model, and settings saved with the name '{custom_name}'")
                else:
                    assistant_response = conversation_agent(user_query, model, corpus, top_k=5, max_tokens=2048)
                    print(f"Assistant: {assistant_response}")
            
            return  # Exit the function after the conversation loop
        else:
            print("Saved data not found.")
            return
    else:
        chunks_with_paths = list(read_and_preprocess(file_paths))
        corpus = [chunk for chunk, _ in chunks_with_paths]
        source_file_paths = [source_file_path for _, source_file_path in chunks_with_paths]

        model = Word2Vec(sentences=corpus, vector_size=256, window=10, min_count=1, workers=4)
        chunk_embeddings = [get_chunk_vector(model, chunk) for chunk in corpus]


    # Perform clustering and graph construction
    distance_matrix = pdist(chunk_embeddings, 'euclidean')
    Z = linkage(distance_matrix, 'ward')
    clusters = fcluster(Z, t=16, criterion='maxclust')
    H = nx.DiGraph()

    for i, cluster_id in enumerate(clusters):
        parent_node = f"Cluster_{cluster_id}"
        child_node = f"Doc_{i}"
        
        H.add_node(parent_node)
        if i < len(source_file_paths):
            source_file_path = source_file_paths[i]
            source_file_name = os.path.basename(source_file_path)
            H.add_node(child_node, text=' '.join(corpus[i]), cluster=cluster_id, source_file=source_file_name)
        else:
            H.add_node(child_node, text=' '.join(corpus[i]), cluster=cluster_id)
        H.add_edge(parent_node, child_node)

    # Perform summarization and graph updates
    cluster_summaries = cluster_summarization(H, model, source_file_paths)
    enriched_cluster_summaries = [enrich_summary(summary, chunk_embeddings, corpus) for summary in cluster_summaries]
    convolutional_summaries, depth = convolutional_summarization(enriched_cluster_summaries, 0)
    flattened_convolutional_summaries = [summary for depth_summaries in convolutional_summaries for summary in depth_summaries]
    enriched_convolutional_summaries = [enrich_summary(summary, chunk_embeddings, corpus) for summary in flattened_convolutional_summaries]
    final_summary = thoughtful_summarization(enriched_convolutional_summaries[-1], depth)

    output_data = {
        'cluster_summaries': cluster_summaries,
        'enriched_cluster_summaries': enriched_cluster_summaries,
        'convolutional_summaries': convolutional_summaries,
        'enriched_convolutional_summaries': enriched_convolutional_summaries,
        'final_summary': final_summary
    }
    json_filename = f"summaries_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    save_to_json(output_data, json_filename)

    summary_sentences = []
    for depth_summaries in convolutional_summaries:
        for summary in depth_summaries:
            summary_text = summary['text']
            summary_doc = nlp(summary_text)
            for sent in summary_doc.sents:
                summary_sentences.append(sent.text.capitalize())

    # Add the summary sentences to the corpus
    updated_corpus = [' '.join(chunk) if isinstance(chunk, list) else chunk for chunk in corpus] + summary_sentences

    # Retrain the Word2Vec model with the updated corpus
    updated_model = Word2Vec(sentences=[gensim.utils.simple_preprocess(sentence) for sentence in updated_corpus], vector_size=256, window=10, min_count=1, workers=4)

    # Calculate the updated chunk embeddings
    updated_chunk_embeddings = [get_chunk_vector(updated_model, gensim.utils.simple_preprocess(sentence)) for sentence in updated_corpus]

    # Conversation loop
    print("Conversation Loop:")
    while True:
        user_query = input("User: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        elif user_query.lower().startswith('save '):
            custom_name = user_query.split(' ', 1)[1]
            save_embeddings(updated_chunk_embeddings, custom_name)
            save_model(updated_model, custom_name)
            save_settings({
                'corpus': updated_corpus,
                'source_file_paths': source_file_paths,
                'vector_size': 256,
                'window': 10,
                'min_count': 1,
                'workers': 4
            }, custom_name)
            print(f"Embeddings, model, and settings saved with the name '{custom_name}'")
        else:
            assistant_response = conversation_agent(user_query, updated_model, updated_corpus, top_k=5, max_tokens=2048)
            print(f"Assistant: {assistant_response}")

if __name__ == "__main__":
    main()
