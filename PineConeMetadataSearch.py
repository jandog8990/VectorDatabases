from sentence_transformers import SentenceTransformer
from dotenv import dotenv_values
import pinecone
import time
import pickle

# PineCone keys for connecting 
config = dotenv_values(".env")
ENV_KEY = config["PINE_CONE_ENV_KEY"] 
API_KEY = config["PINE_CONE_API_KEY"] 
pinecone.init(
    api_key=API_KEY,
    environment=ENV_KEY
)

# load dataset from pickle file
with open('squad.pkl', 'rb') as f:
    dataset = pickle.load(f) 
print(f"Dataset len from pickle = {len(dataset)}")
print("\n")

# create ids from the results
def create_ids(results):
    # get the ids from the results
    ids = [obj['id'] for obj in results['matches']]
    print(f"Ids len = {len(ids)}")
    print(ids)
    print("\n")
    return ids

# get samples from the main dataset
def get_samples(dataset):
    return {
        data['id']: {
            'context': data['context'],
            'metadata': data['metadata']
        } for data in dataset} 

# show results from query
dataset_samples = get_samples(dataset)
def show_results(results):
    ids = create_ids(results)
    for i in ids:
        print(f"Get sample id = {i}") 
        print(dataset_samples[i])

# create the pincone index
#pinecone.create_index(name='squad-test', metric='euclidean', dimension=768)
index = pinecone.Index('squad-test')
print("Index:")
print(index)
print("\n")

# let's test the embedder
search_query = "Early engineering courses provided by American Universities in the 1870s."
model = SentenceTransformer('stsb-xlm-r-multilingual')
query = model.encode(search_query)

# time the results for the query
start = time.time()
results = index.query(vector=[query.tolist()], top_k=3)
end = time.time()
total = end - start

# load the query squad dataset
print("Results from regular query:")
show_results(results)
print("\n")

# filter searching using conditions
# $nin = not in array
conditions = {
    'lang': {'$eq': 'en'},
    'title': {'$nin': ['University_of_Kansas', 'University_of_Notre_Dame']}
}
results = index.query(vector=[query.tolist()], top_k=3, filter=conditions)
print("Results from conditional query:")
show_results(results)
print("\n")

