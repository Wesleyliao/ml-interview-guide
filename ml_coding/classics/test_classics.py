import importlib.util
from pathlib import Path
import numpy as np
import pytest

def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_linear_rank_deficient():
    X=np.array([[0,0],[1,1],[2,2],[3,3]],float)
    y=2+3*X[:,0]
    model=load('linear_regression').LinearRegressor().fit(X,y)
    np.testing.assert_allclose(model.predict(X),y,atol=1e-10)

def test_kmeans_empty_and_separated():
    cls=load('kmeans').Kmeans
    m=cls(3).fit(np.ones((4,2)))
    assert np.isfinite(m.centroids).all() and m.inertia_==0
    X=np.array([[0,0],[0,1],[10,10],[10,11]],float)
    m=cls(2).fit(X)
    labels=m.predict(X)
    assert labels[0]==labels[1] and labels[2]==labels[3] and labels[0]!=labels[2]

def test_nb_unknown_and_refit():
    m=load('naive_bayes').NaiveBayes().fit([['red'],['red'],['blue']],[1,1,0])
    assert m.predict([['red'],['new']])==[1,1]
    m.fit([['blue']],[0])
    assert m.predict([['red']])==[0]

def test_attention_shape_gradient_and_formula():
    torch=pytest.importorskip('torch')
    m=load('self_attention').SelfAttention(4,3,2)
    X=torch.randn(2,5,4,requires_grad=True)
    out=m(X)
    expected=torch.softmax(m.Wq(X)@m.Wk(X).transpose(-1,-2)/3**0.5,-1)@m.Wv(X)
    torch.testing.assert_close(out,expected)
    assert out.shape==(2,5,2)
    out.sum().backward()
    assert torch.isfinite(X.grad).all()

def test_numpy_attention_matches_matmul_and_single_token():
    rng = np.random.default_rng(7)
    Wq, Wk = rng.normal(size=(2,4,3))
    Wv = rng.normal(size=(4,2))
    model = load('self_attention_numpy').SelfAttentionNumpy(Wq,Wk,Wv)
    X = rng.normal(size=(2,5,4))
    scores = (X@Wq) @ (X@Wk).transpose(0,2,1) / np.sqrt(3)
    weights = np.exp(scores-scores.max(axis=-1,keepdims=True))
    expected = (weights/weights.sum(axis=-1,keepdims=True)) @ (X@Wv)
    np.testing.assert_allclose(model.forward(X),expected,atol=1e-12)
    np.testing.assert_allclose(model.forward(X[:,:1]),X[:,:1]@Wv)

def test_transformer_blocks_match_without_biases():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(19)
    E, H = 4, 7
    Wq, Wk, Wv = rng.normal(scale=0.2, size=(3,E,E))
    W1 = rng.normal(scale=0.2, size=(E,H))
    W2 = rng.normal(scale=0.2, size=(H,E))
    X = rng.normal(size=(2,3,E))
    original = X.copy()
    numpy_block = load('self_attention_numpy').TransformerBlockNumpy(Wq,Wk,Wv,W1,W2)
    torch_block = load('self_attention').TransformerBlock(E,H).double()
    with torch.no_grad():
        for layer, weight in [(torch_block.attention.Wq,Wq), (torch_block.attention.Wk,Wk),
                              (torch_block.attention.Wv,Wv), (torch_block.ffn[0],W1),
                              (torch_block.ffn[2],W2)]:
            layer.weight.copy_(torch.from_numpy(weight.T))
            layer.bias.zero_()
    tensor = torch.tensor(X, requires_grad=True)
    result = torch_block(tensor)
    np.testing.assert_allclose(numpy_block.forward(X),result.detach().numpy(),atol=1e-10)
    np.testing.assert_array_equal(X, original)
    result.sum().backward()
    assert torch.isfinite(tensor.grad).all()
    # Constant token features exercise the normalization epsilon.
    assert np.isfinite(numpy_block.forward(np.ones((1,2,E)))).all()
