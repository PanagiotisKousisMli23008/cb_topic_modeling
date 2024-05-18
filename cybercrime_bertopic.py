# Install bertopic
!pip install bertopic

# Data processing
import pandas as pd
import numpy as np
# Text preprocessing
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
# Topic model
from bertopic import BERTopic
# Dimension reduction
from umap import UMAP

# Read in data
abstracts = pd.read_csv('cybercrime.txt', sep='\t', names=['abstract'])
# Take a look at the data
abstracts.head()

# Get the dataset information
abstracts.info()

# Remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
print(f'There are {len(stopwords)} default stopwords. They are {stopwords}')

# Remove stopwords
abstracts['abstracts_without_stopwords'] = abstracts['abstract'].apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in stopwords]))
# Lemmatization
abstracts['abstracts_lemmatized'] = abstracts['abstracts_without_stopwords'].apply(lambda x: ' '.join([wn.lemmatize(w) for w in x.split() if w not in stopwords]))
# Take a look at the data
abstracts.head()

# Initiate UMAP
umap_model = UMAP(n_neighbors=15,
                  n_components=5,
                  min_dist=0.0,
                  metric='cosine',
                  random_state=100)
# Initiate BERTopic
topic_model = BERTopic(umap_model=umap_model, language="english", calculate_probabilities=True)
# Run BERTopic model
topics, probabilities = topic_model.fit_transform(abstracts['abstracts_lemmatized'])

# Get the list of topics
topic_model.get_topic_info()

# Visualize top topic keywords
topic_model.visualize_barchart(top_n_topics=70)

topicsfig = topic_model.visualize_barchart(top_n_topics=70)
topicsfig.write_html('cybercrime_topics.html')

# Visualize term rank decrease
topic_model.visualize_term_rank()

# Visualize intertopic distance
topic_model.visualize_topics(top_n_topics=70)

topicsfig = topic_model.visualize_topics()
topicsfig.write_html('cybercrime_intertopic_distance.html')

# Visualize connections between topics using hierachical clustering
topic_model.visualize_hierarchy(top_n_topics=70)

topicsfig = topic_model.visualize_hierarchy()
topicsfig.write_html('cybercrime_hierarchy.html')

# Visualize similarity using heatmap
topic_model.visualize_heatmap(top_n_topics=50)

topicsfig = topic_model.visualize_heatmap()
topicsfig.write_html('cybercrime_heatmap.html')

# Visualize probability distribution
topic_model.visualize_distribution(topic_model.probabilities_[0], min_probability=0.010)

chart = topic_model.visualize_distribution(topic_model.probabilities_[0], min_probability=0.01)
# Write the chart as a html file
chart.write_html("cybercrime_topic_probability_distribution.html")

# Check the content for the first review
abstracts['abstract'][1]

# Get probabilities for all topics
topic_model.probabilities_[0]

# Get the topic predictions
topic_prediction = topic_model.topics_[:]
# Save the predictions in the dataframe
abstracts['topic_prediction'] = topic_prediction
# Take a look at the data
abstracts.head(50)

# New data
new_abstract = "we will talk about smart grid. Smart grid is an electricity network for energy efficiancy"
# Find topics
num_of_topics = 3
similar_topics, similarity = topic_model.find_topics(new_abstract, top_n=num_of_topics);
# Print results
print(f'The top {num_of_topics} similar topics are {similar_topics}, and the similarities are {np.round(similarity,2)}')

# Print the top keywords for the top similar topics
for i in range(num_of_topics):
  print(f'The top keywords for topic {similar_topics[i]} are:')
  print(topic_model.get_topic(similar_topics[i]))
  
  #find representive docs
for topic in range (51):
  print('Representive abstracts for topic', topic)
  docs=topic_model.get_representative_docs(topic)
  print(docs)
  
docs=abstracts['abstracts_lemmatized']

topics_to_merge = [[-1, 14, 24, 29, 37, 39, 46, 48], [0, 35], [1, 13, 19, 27, 42], [2, 28, 33], [3, 10, 20], [4, 34, 40, 54], [5, 11, 21, 23, 31, 35, 36, 43, 52, 57], [6, 12, 22, 55], [7, 8, 58], [9, 26, 45], [15, 18, 38, 56], [16, 50], [25,51], [32,49], [44, 47]]

topic_model.merge_topics(docs=docs, topics_to_merge=topics_to_merge)

merged_topic_model = topic_model

topic_model.get_topic_info()

labels = ['-1', 'Cyberbullying', 'Dark web', 'Digital evidence', 'Policing', 'Malware', 'Phishing', 'IoT', 'Cybersecurity', 'Law', 'Network', 'Education', 'Helathcare', 'Ransomware', 'Cyberfraud', 'Cryptography', 'Data mining', 'Banking', 'Mobile']

merged_topic_model.set_topic_labels(labels)

#topic_model = BERTopic.load('bert_model_merged')
merged_topic_model.get_topic_info()

merged_topic_model.visualize_barchart(top_n_topics=52 ,custom_labels=True)
chart = merged_topic_model.visualize_barchart(top_n_topics=52 ,custom_labels=True)
# Write the chart as a html file
chart.write_html("abstracts_barchart_merged.html")

merged_topic_model.visualize_topics()
chart = merged_topic_model.visualize_topics()
# Write the chart as a html file
chart.write_html("abstracts_topics_merged.html")

merged_topic_model.visualize_documents(docs=docs, hide_annotations=False, custom_labels=True)
chart = merged_topic_model.visualize_documents(docs=docs, hide_annotations=False, custom_labels=True)
# Write the chart as a html file
chart.write_html("abstracts_docviz_merged.html")

merged_topic_model.visualize_hierarchy(custom_labels=True)
chart = merged_topic_model.visualize_hierarchy(custom_labels=True)
# Write the chart as a html file
chart.write_html("abstracts_hierarchy_merged.html")

merged_topic_model.visualize_heatmap(custom_labels=True)
chart = merged_topic_model.visualize_heatmap(custom_labels=True)
chart.write_html("abstracts_heatmap_merged.html")

hierarchical_topics = merged_topic_model.hierarchical_topics(docs)
tree = topic_model.get_topic_tree(hierarchical_topics)
print(tree)

#find representive docs
for topic in range (18):
  print('Representive abstracts for topic', topic)
  docs=merged_topic_model.get_representative_docs(topic)
  print(docs)
