import networkx as nx 
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
# from huggingface_hub import InferenceClient
import numpy as np
from scipy.linalg import expm
import random
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from openai import OpenAI
from evaluate import load
from datasets import load_dataset
import os
os.environ["OPENAI_API_KEY"] = ""

# ----------------- Node creation ----------------- #
class OllamaNode:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def activate(self, predecessor, user):
        system_prompt = "You are a node in a network of LLMs. Your purpose is to incrementelly improve and refine the response to the user's query. You can do this, for example, by adding more information to the replies of your predecessors, fact checking or refining the answer. Also make sure that the user cannot distinguish you from the network."
        full_message = f"System: {system_prompt} \n Answer from your predecessor(s): {predecessor} \n Original user query: {user}"
        AIReply = ChatOllama(model=self.model_name).invoke(full_message)
        content = AIReply.content
        total_tokens = AIReply.usage_metadata["total_tokens"]
        return content, total_tokens
    
class OpenAINode:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def activate(self, predecessor, user):
        system_prompt = "You are a node in a network of LLMs. Your purpose is to incrementelly improve and refine the response to the user's query. You can do this, for example, by adding more information to the replies of your predecessors, fact checking or refining the previous answer."
        chatCompletion = self.client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": f"{system_prompt}",
                "role": "assistant",
                "content": f"{predecessor}",
                "role": "user",
                "content": f"{user}",}],
            model=f"{self.model_name}",
        )

        response = chatCompletion.choices[0].message.content
        tokens_used = chatCompletion.usage.total_tokens

        return response, tokens_used                               

tinyllama = OllamaNode("tinyllama")
llama32 = OllamaNode("llama3.2")
gpt4o = OpenAINode("gpt-4o-mini")


# ----------------- Feasibility check ----------------- #
def find_sink_nodes(A):
    # Return indices of nodes that have no outgoing edges.
    return [i for i in range(A.shape[0]) if np.all(A[i, :] == 0)]

def has_single_sink(A):
    sinks = find_sink_nodes(A)
    if len(sinks) == 1:
        return True
    else:
        return False

def is_acyclic(A):
    # Check if the graph has any cycles.
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(G)

def is_feasible(A):
    if has_single_sink(A) and is_acyclic(A):
        return True
    else:
        return False
    

# ----------------- Graph execution ----------------- #
def execute_graph(G: nx.DiGraph, user_query):
    topological_order = list(nx.topological_sort(G))

    node_outputs = {}

    total_consumed_tokens = 0

    for node in topological_order:
        # If the node has no incoming edges, it gets a special input.
        if G.in_degree(node) == 0:
            predecessor_outputs = "There are no predecessor nodes. You are first!"
        else:
            # Gather outputs from immediate predecessors and join them.
            predecessor_outputs =  "\n".join([node_outputs[predecessor] for predecessor in G.predecessors(node)])

        # Call model and save the content and total tokens.
        LLM_data = node.activate(predecessor = predecessor_outputs, user = user_query)

        # Pick out the message intended for the next node
        node_outputs[node] = LLM_data[0]

        # Update the total consumed tokens
        total_consumed_tokens += LLM_data[1]

    sink_node = topological_order[-1]
    final_answer = node_outputs[sink_node]
    return final_answer, total_consumed_tokens

# ----------------- Evaluation module ----------------- #
gsm8k = load_dataset("gsm8k", "main")  # GSM8K dataset from Hugging Face
train_data = gsm8k["train"]
bertscore = load("bertscore")

class BERTscorer:
    def __init__(self, dataset, LLM_graph: nx.DiGraph):
        self.dataset = dataset
        self.G = LLM_graph

    def evaluate(self, N = 5):
        random_questions = random.sample(list(self.dataset), N)
        predictions = []
        references = [entry["answer"] for entry in random_questions]
        total_tokens = 0

        for entry in random_questions:
            graph_prediction, tokens = execute_graph(self.G, entry["question"])
            predictions.append(graph_prediction)
            total_tokens += tokens
        score = bertscore.compute(predictions = predictions, references = references, lang='en')
        return np.mean(score['f1']), total_tokens
    

# ----------------- Kernel function ----------------- #

def symmetrize_laplacian(A):
    # Symmetrize the adjacency matrix
    A_sym = (A + np.transpose(A)) / 2.0
    # Compute the degree matrix (diagonal) from the symmetrized adjacency matrix
    degrees = np.sum(A_sym, axis=1)
    D_sym = np.diag(degrees)
    # Compute the symmetric Laplacian
    L_sym = D_sym - A_sym
    return L_sym

def laplacian_kernel(A1, A2, sigma=0.5):
    # Compute the symmetric Laplacians for both graphs
    L1 = symmetrize_laplacian(A1)
    L2 = symmetrize_laplacian(A2)
    
    # Compute the Frobenius norm squared of the difference
    diff = L1 - L2
    fro_norm_sq = np.linalg.norm(diff, 'fro')**2
    
    # Compute and return the kernel value using the Gaussian (RBF) form
    k = np.exp(-fro_norm_sq / sigma**2)
    return k

# ----------------- Kernel Matrix calculation ----------------- #
def calc_new_kernel_matrix(graphs):
    num_graphs = np.shape(graphs)[0]
    kernel_matrix = np.zeros((num_graphs, num_graphs))

    for i in range(num_graphs):
        for j in range(num_graphs):
            kernel_matrix[i,j] = laplacian_kernel(graphs[i], graphs[j])

    return kernel_matrix


# ----------------- Acquisition function ----------------- #
def ucb(graph, kernel_matrix, previous_graphs_adj_matrices, y_values, t, num_of_possible_graphs, delta):
    if is_feasible(graph):
        beta_t = 2*np.log(num_of_possible_graphs*t**2 * np.pi**2/(6*delta))
        k = []
        for adj_matrix in previous_graphs_adj_matrices:
            k.append(laplacian_kernel(graph, adj_matrix))

        k_column = np.array(k).reshape(-1,1)
        y_column = np.array(y_values).reshape(-1,1)

        mu = np.dot(np.dot(k, np.linalg.inv(kernel_matrix)), y_column)
        sigma = np.sqrt(1 - np.dot(np.dot(k, np.linalg.inv(kernel_matrix)) , k_column))

        ucb_value = mu + np.sqrt(beta_t)*sigma
        return ucb_value.item()
    else:
        # Penalises infeasible graphs in order to enforce the constraints.
        return -100000
    

# ----------------- Objective function definition ----------------- #
def objective_function(graph: nx.DiGraph, weights):
    w1 = weights[0]
    w2 = weights[1]

    scorer = BERTscorer(train_data, graph)
    f1_score, used_tokens = scorer.evaluate(N=7)
    
    return w1*f1_score - w2*used_tokens

# ----------------- Probabilistic Reparameterization framework ----------------- #
def int_to_adj_matrix(i, N):
    i = int(i)
    bitstring = format(i, f'0{N**2}b')
    bits = np.array([int(b) for b in bitstring], dtype=np.int8)
    return bits.reshape(N, N)

class ProbabilisticReparameterization:
    def __init__(self, num_nodes, num_categories):
        self.num_categories = num_categories
        self.phi = torch.nn.Parameter(torch.randn(num_categories), requires_grad=True)
        self.optimizer = optim.Adam([self.phi], lr=0.1)
        self.tau = 0.7
        self.num_nodes = num_nodes

    def sample(self, num_samples=10):
        theta = torch.nn.functional.softmax((self.phi - 0.5) / self.tau, dim=0)
        distribution = torch.distributions.Categorical(theta)
        samples = distribution.sample((num_samples,))
        log_probs = distribution.log_prob(samples) 
        return samples, log_probs

    def optimize_acquisition(self, kernel_matrix, previous_graphs_adj_matrices, y_values, t, results_df, num_steps=3, num_samples=2):
        tau = self.tau
        best_ucb_mean = None
        best_ucb_array = None
        best_samples = None

        for step in range(num_steps):
            self.optimizer.zero_grad()
            
            # Sample discrete values
            z_samples, log_probs = self.sample(num_samples)
            # Compute acquisition function for each sample
            ucb_values = [ucb(int_to_adj_matrix(z, self.num_nodes), kernel_matrix, previous_graphs_adj_matrices, y_values, t, self.num_categories, delta=0.1) for z in z_samples]

            # Compute expectation using Monte Carlo estimator
            UCB_expecation_MC_estimator = np.mean(ucb_values)

            if best_ucb_mean is None or UCB_expecation_MC_estimator > best_ucb_mean:
                best_ucb_mean = UCB_expecation_MC_estimator
                best_ucb_array = ucb_values
                best_samples = z_samples                

            # Compute the MC estimator of the gradient

            MC_gradient_estimator = torch.mean(-(log_probs * torch.tensor(ucb_values)))
            MC_gradient_estimator.backward()
            self.optimizer.step()
        
            # Decay temperature tau
            tau = tau * 0.99
        best_sample = best_samples[torch.argmax(torch.tensor(best_ucb_array))]
        return best_sample, results_df

# ----------------- BO initialization ----------------- #
nodes = [gpt4o, llama32, tinyllama]

G1 = nx.DiGraph()
G1.add_nodes_from(nodes)
G1.add_edge(tinyllama, gpt4o)
G1.add_edge(gpt4o, llama32)

G2 = nx.DiGraph()
G2.add_nodes_from(nodes)
G2.add_edge(gpt4o, llama32)
G2.add_edge(llama32, tinyllama)

y1 = objective_function(G1, [1000, 0.01])
y2 = objective_function(G2, [1000, 0.01])

# The nodelist argument is crucial as we need the adjacency matrices to have the same structure. Otherwise they will not encode similarity!
adj_matrices = [nx.to_numpy_array(G1, nodelist=nodes), nx.to_numpy_array(G2, nodelist=nodes)]
y_values = [y1, y2]

# ----------------- Main optimization loop ----------------- #
def BayesianOptimisationLoop(adj_matrices, y_values, objective_function_variance, numIter = 5):
    columns = ["t", "graph_index", "AF_value"]
    results_df = pd.DataFrame(columns=columns)
    
    # Defining the weights used to adjust the relationship between tokens used and the f1 score from the objective function.
    w_score = 1000
    w_tokens = 0.01

    # The number of possible graphs
    num_nodes = len(nodes)
    num_categories = 2**(num_nodes*num_nodes)

    # Initialise the PR class
    probReparam = ProbabilisticReparameterization(num_nodes, num_categories)
    
    t = 1
    while t < numIter + 1:
        # Calculate the kernel matrix (i.e. fit the GP model with the data)
        kernel_matrix = calc_new_kernel_matrix(adj_matrices)
        np.fill_diagonal(kernel_matrix, kernel_matrix.diagonal() + objective_function_variance)
 
        # Use PR method to optimise the AF in order to find our most interesting next point
        next_graph, results_df = probReparam.optimize_acquisition(kernel_matrix, adj_matrices, y_values, t, results_df, num_steps=50, num_samples=5)
        
        # Convert the adjacency matrix to a nx DiGraph to be able to evaluate it
        next_graph = int_to_adj_matrix(int(next_graph), num_nodes)

        G = nx.from_numpy_array(next_graph, create_using=nx.DiGraph, nodelist=nodes)

        # Evaluate the graph on the objective function
        next_y_value = objective_function(G, [w_score, w_tokens])

        # Add the adjacency matrix and the score to the database
        adj_matrices.append(next_graph)
        y_values.append(next_y_value)

        print(f"Round: {t}")
        print(f"Evaluated score: {next_y_value}")
        t += 1
    best_solution = np.argmax(y_values)
    return adj_matrices[best_solution], y_values, results_df

# ----------------- Run the optimization loop ----------------- #
optimal_graph, f_values, results_df = BayesianOptimisationLoop(adj_matrices, y_values, objective_function_variance = 242.3573890907673, numIter = 10)

print(f"The highest score was {max(f_values)}.")

# ----------------- Visualize the optimized graph ----------------- #
G = nx.from_numpy_array(optimal_graph, create_using=nx.DiGraph, nodelist=nodes)

# Relabel nodes with model names
mapping = {node: node.model_name for node in nodes}
G = nx.relabel_nodes(G, mapping)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=5500, node_color='skyblue', font_size=12, font_family='sans-serif', edge_color='gray', arrows=True, arrowsize=20)
plt.show()