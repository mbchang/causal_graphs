from collections import namedtuple
import copy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from plot_utils import plot3d, AxisInfo, PlotInfo, Axis3dInfo

subroot = 'debug'
if not os.path.exists(subroot):
    os.mkdir(subroot)

Node = namedtuple('Node', ('id', 'dist', 'dim'))
SphericalMultivariateNormalParams = namedtuple('SphericalMultivariateNormalParams', ('mu', 'logstd'))
GraphSpecification = namedtuple('GraphSpecification', ('node_ids', 'edge_ids'))
SphericalMultivariateNormalParam = namedtuple('SphericalMultivariateNormalParam', ('name', 'value'))
Dataset = namedtuple('Dataset', ('inputs', 'outputs', 'info'))
AdjacencyValue = namedtuple('AdjacencyValue', ('incoming_edge_id', 'outgoing_edge_ids'))

def visualize_parameters(model, name):
    for n, p in model.named_parameters():
        if p.grad is None:
            print('{}-{}:\t{}\t{}'.format(name, n, p.data.norm(), None))
        else:
            print('{}-{}:\t{}\t{}'.format(name, n, p.data.norm(), p.grad.data.norm()))

def gradnorm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

class SphericalMultivariateNormal(MultivariateNormal):
    def __init__(self, mu, logstd):
        MultivariateNormal.__init__(self, loc=mu, scale_tril=torch.diag_embed(torch.exp(logstd)))

def root_edge(to_node):
    mu = nn.Parameter(torch.zeros(to_node.dim))
    logstd = nn.Parameter(torch.zeros(to_node.dim))
    edge = Root(to_node=to_node, mu=mu, logstd=logstd)
    return edge

def define_edge(from_node, to_node):
    mu = nn.Linear(from_node.dim, to_node.dim)
    logstd = nn.Parameter(torch.zeros(to_node.dim))
    edge = Edge(from_node=from_node, to_node=to_node, mu=mu, logstd=logstd)
    return edge

def modify_edge(edge, param):
    new_edge = copy.deepcopy(edge)
    new_edge.set_param(param.name, param.value)
    return new_edge

def check_valid_specification(graph):
    """
        For every node, make sure that there exists exactly one edge that produces it
    """
    for node_name, node in graph.nodes.items():
        found_associated_edge = False
        for edge_name, edge in graph.edges.items():
            if edge.to_node.id == node.id:
                assert found_associated_edge == False
                found_associated_edge = True
        assert found_associated_edge == True

class SpaceOfGraphs():
    def __init__(self):
        self.nodes = self.define_nodes()
        self.edges = self.define_edges(self.nodes)

    def define_nodes(self):
        """
            returns a dictionary of {node_id: Node}
        """
        raise NotImplementedError

    def define_edges(self, nodes):
        """
            returns a dictionary of {edge_id: Edge}
        """
        raise NotImplementedError

    def select_graph(self, specification):
        nodes = dict()
        for node_id in specification.node_ids:
            nodes[node_id] = self.nodes[node_id]
        edges = dict()
        for edge_id in specification.edge_ids:
            edges[edge_id] = self.edges[edge_id]
        graph = Graph(nodes=nodes, edges=edges)
        check_valid_specification(graph)
        return graph

class TestSpaceOfGraphs(SpaceOfGraphs):
    def __init__(self, x2_dim=1, x2_std=1):
        self.x2_dim = x2_dim
        self.x2_std = x2_std
        SpaceOfGraphs.__init__(self)

    def define_nodes(self):
        x1 = Node(id='x1', dim=1,
            dist=lambda dparams: SphericalMultivariateNormal(
                mu=dparams.mu, logstd=dparams.logstd),
            )

        x2 = Node(id='x2', dim=self.x2_dim,
            dist=lambda dparams: SphericalMultivariateNormal(
                mu=dparams.mu, logstd=dparams.logstd),
            )

        y = Node(id='y', dim=1,
            dist=lambda dparams: SphericalMultivariateNormal(
                mu=dparams.mu, logstd=dparams.logstd),
            )

        z = Node(id='z', dim=1,
            dist=lambda dparams: SphericalMultivariateNormal(
                mu=dparams.mu, logstd=dparams.logstd),
            )

        # need the keys of the dictionary to be the same as the node id
        nodes = {node.id: node for node in [x1, x2, y, z]} 
        return nodes

    def define_edges(self, nodes):
        """
        At this point we are assuming we know the distribution type
        """
        root_x1 = root_edge(nodes['x1'])
        root_z = root_edge(nodes['z'])
        x1_x2 = define_edge(nodes['x1'], nodes['x2'])
        x1_y = define_edge(nodes['x1'], nodes['y'])
        z_x1 = define_edge(nodes['z'], nodes['x1'])
        z_x2 = define_edge(nodes['z'], nodes['x2'])
        y_x2 = define_edge(nodes['y'], nodes['x2'])

        x1_x2_diff_func = modify_edge(x1_x2, 
            SphericalMultivariateNormalParam(name='mu',value=nn.Linear(nodes['x1'].dim, nodes['x2'].dim)))
        x1_x2_diff_noise = modify_edge(x1_x2, 
            SphericalMultivariateNormalParam(name='logstd',value=nn.Parameter(torch.empty(nodes['x2'].dim).fill_(np.log(self.x2_std)))))
        y_x2_diff_func = modify_edge(y_x2, 
            SphericalMultivariateNormalParam(name='mu',value=nn.Linear(nodes['y'].dim, nodes['x2'].dim)))
        y_x2_diff_noise = modify_edge(y_x2, 
            SphericalMultivariateNormalParam(name='logstd',value=nn.Parameter(torch.empty(nodes['x2'].dim).fill_(np.log(self.x2_std)))))

        edges = dict(
            root_x1=root_x1,
            root_z=root_z,
            x1_x2=x1_x2, 
            x1_y=x1_y, 
            z_x1=z_x1, 
            z_x2=z_x2,
            x1_x2_diff_func=x1_x2_diff_func,
            x1_x2_diff_noise=x1_x2_diff_noise,
            y_x2=y_x2,
            y_x2_diff_func=y_x2_diff_func,
            y_x2_diff_noise=y_x2_diff_noise
            )

        # create more edges with varying dim and std
        edges_std = {
            'y_x2_std{}'.format(std): modify_edge(y_x2, 
            SphericalMultivariateNormalParam(name='logstd',value=nn.Parameter(torch.empty(nodes['x2'].dim).fill_(np.log(std)))))
            for std in range(1, 51, 2)
        }

        edges = {**edges, **edges_std}

        return edges

class Edge():
    def __init__(self, from_node, to_node, mu, logstd):
        self.from_node = from_node
        self.to_node = to_node
        self.mu = mu
        self.logstd = logstd

    def __call__(self, from_node_sample):
        return SphericalMultivariateNormalParams(mu=self.mu(from_node_sample),logstd=self.logstd)

    def set_param(self, param_name, param_value):
        if param_name == 'mu':
            self.mu = param_value
        elif param_name == 'logstd':
            self.logstd = param_value
        else:
            assert False

class Root():
    def __init__(self, to_node, mu, logstd):
        self.to_node = to_node
        self.mu = mu
        self.logstd = logstd

    def __call__(self):
        return SphericalMultivariateNormalParams(mu=self.mu, logstd=self.logstd)

    def set_param(self, param_name, param_value):
        if param_name == 'mu':
            self.mu = param_value
        elif param_name == 'logstd':
            self.logstd = param_value
        else:
            assert False

class Graph():
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.adjacency_dict, self.root_id = self.build_adjacency_dict(self.nodes, self.edges)

    def build_adjacency_dict(self, nodes, edges):
        adjacency_dict = dict()

        def build_adjacency_dict_recursive(current_node, incoming_edge_id, adjacency_dict):
            outgoing_edge_condition = lambda edge: not isinstance(edge, Root) and edge.from_node == current_node
            outgoing_edge_ids = sorted([e_id for e_id in edges if outgoing_edge_condition(edges[e_id])])

            adjacency_dict[current_node.id] = AdjacencyValue(
                incoming_edge_id=incoming_edge_id,
                outgoing_edge_ids=outgoing_edge_ids
                )

            for outgoing_edge_id in outgoing_edge_ids:
                outgoing_node = edges[outgoing_edge_id].to_node
                adjacency_dict = build_adjacency_dict_recursive(
                    current_node=outgoing_node,
                    incoming_edge_id=outgoing_edge_id,
                    adjacency_dict=adjacency_dict)
            return adjacency_dict

        root_edge_id = filter_assert_one(lambda e_id: isinstance(edges[e_id], Root), edges)
        root_node = edges[root_edge_id].to_node  # but actually we can have multiple roots
        adjacency_dict = build_adjacency_dict_recursive(
            current_node=root_node,
            incoming_edge_id=root_edge_id,
            adjacency_dict=adjacency_dict,
            )
        return adjacency_dict, root_node.id

    def sample(self, data_size):
        with torch.no_grad():
            samples = dict()

            def sample_recursive(current_node_id, samples_so_far):
                current_node = self.nodes[current_node_id]
                incoming_edge_id = self.adjacency_dict[current_node_id].incoming_edge_id
                if isinstance(self.edges[incoming_edge_id], Root):  # actually we can have multiple roots
                    dparams = self.edges[incoming_edge_id]()
                    current_sample = current_node.dist(dparams).sample(torch.Size([data_size]))
                else:
                    parent_node_id = self.edges[incoming_edge_id].from_node.id
                    parent_sample = samples_so_far[parent_node_id]
                    dparams = self.edges[incoming_edge_id](parent_sample)
                    current_sample = current_node.dist(dparams).sample()
                assert current_node_id not in samples_so_far
                samples_so_far[current_node_id] = current_sample

                for outgoing_edge_id in self.adjacency_dict[current_node_id].outgoing_edge_ids:
                    outgoing_node_id = self.edges[outgoing_edge_id].to_node.id
                    sample_recursive(outgoing_node_id, samples_so_far)
                return samples_so_far

            samples = sample_recursive(self.root_id, samples)
            return samples

def filter_assert_one(condition, lst):
    matches = [x for x in lst if condition(x)]
    assert len(matches) == 1
    return matches[0]

def optimize_model(model, dataset):
    num_epochs = 1000
    lr = 0.001

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epochs):
        output = model(dataset.inputs)
        objective_value = F.mse_loss(output, dataset.outputs)
        optimizer.zero_grad()
        objective_value.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print('Epoch {} | Train Loss: {}\tGradient Norm:{}'.format(
                epoch, objective_value.item(), gradnorm(model)))

    return model

def evaluate_model(model, test_dataset):
    with torch.no_grad():
        output = model(test_dataset.inputs)
        objective_value = F.mse_loss(output, test_dataset.outputs)
    print('Test Loss: {}'.format(objective_value))
    return objective_value

def create_dataset(graph, inp_vars, out_vars, data_size):
    samples = graph.sample(data_size)
    dataset = Dataset(
        inputs=torch.cat([samples[node_id] for node_id in inp_vars], dim=-1),
        outputs=torch.cat([samples[node_id] for node_id in out_vars], dim=-1),
        info=dict(graph=graph))
    return dataset


def create_dataset(graph, inp_vars, out_vars):
    def create_dataset_with_size(data_size):
        samples = graph.sample(data_size)
        dataset = Dataset(
            inputs=torch.cat([samples[node_id] for node_id in inp_vars], dim=-1),
            outputs=torch.cat([samples[node_id] for node_id in out_vars], dim=-1),
            info=dict(graph=graph))
        return dataset
    return create_dataset_with_size

def run_experiment(dataset_train, datasets_test):
    indim = dataset_train.inputs.shape[-1]
    outdim = dataset_train.outputs.shape[-1]

    torch.manual_seed(0)
    model = nn.Linear(indim, outdim)
    visualize_parameters(model, 'linear')
    optimize_model(model, dataset_train)

    test_losses = dict()
    for test_graph_name, dataset_test in datasets_test.items():
        print('Evaluating on {}'.format(test_graph_name))
        test_loss = evaluate_model(model, dataset_test)
        test_losses[test_graph_name] = test_loss.item()

    return test_losses

def plot(test_losses, fname, title):
    x = []
    y = []
    x_ticks = []
    for i, (test_loss_name, test_loss) in enumerate(test_losses.items()):
        x.append(i)
        x_ticks.append(test_loss_name)
        y.append(test_loss)
    plt.xticks(x, x_ticks, rotation=45)
    plt.plot(x, y)
    plt.ylabel('Test Loss')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(subroot, fname))
    plt.close()


"""
Here define some datasets
"""
def dim_noise_graphs(dim_range, std_range):
    torch.manual_seed(0)
    datasets = dict()
    for dim in dim_range:
        sog = TestSpaceOfGraphs(x2_dim=dim)
        for std in std_range:
            graph = sog.select_graph(GraphSpecification(
                node_ids=['x1', 'x2', 'y'],
                edge_ids=['root_x1', 'y_x2_std{}'.format(std), 'x1_y']))
            datasets['sc_std{}_dim{}'.format(std, dim)] = create_dataset(
                graph=graph,
                inp_vars=['x1', 'x2'],
                out_vars=['y'])
    return datasets

def test():
    data_size = 1000
    dims = range(1, 10, 2)
    stds = range(1, 10, 2)

    all_test_losses = dict()

    datasets = dim_noise_graphs(dims, stds)

    torch.manual_seed(0)
    for dim in dims:
        dataset_train = datasets['sc_std1_dim{}'.format(dim)](data_size=data_size)
        datasets_test = {
            'sc_std{}_dim{}'.format(std, dim): datasets['sc_std{}_dim{}'.format(std, dim)](data_size=data_size) for std in stds
        }
        test_losses = run_experiment(dataset_train, datasets_test)
        all_test_losses[dim] = test_losses
        print(test_losses)
        plot(test_losses, fname='dim{}_noise.png'.format(dim), title='{}-Dim: Varying Noise for Spurious Correlation'.format(dim))
    plot3d(
        data=all_test_losses,
        axis3dinfo=Axis3dInfo(
            x=AxisInfo(range=dims, name='Dim'),
            y=AxisInfo(range=stds, name='Std'),
            z=AxisInfo(range=None, name='Test Loss')),
        plot_info=PlotInfo(
            fname=os.path.join(subroot, 'dim_noise2.png'), 
            title='Varying Noise and Dim for Spurious Correlation'))

if __name__ == '__main__':
    torch.manual_seed(0)
    test()

