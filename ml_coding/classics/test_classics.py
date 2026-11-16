import importlib.util
from pathlib import Path
import numpy as np
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linear_rank_deficient():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float)
    y = 2 + 3 * X[:, 0]
    model = load('linear_regression').LinearRegressor().fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-10)


def test_kmeans_empty_and_separated():
    cls = load('kmeans').Kmeans
    m = cls(3).fit(np.ones((4, 2)))
    assert np.isfinite(m.centroids).all() and m.inertia_ == 0
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], float)
    m = cls(2).fit(X)
    labels = m.predict(X)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]


def test_nb_unknown_and_refit():
    m = load('naive_bayes').NaiveBayes().fit([['red'], ['red'], ['blue']], [1, 1, 0])
    assert m.predict([['red'], ['new']]) == [1, 1]
    m.fit([['blue']], [0])
    assert m.predict([['red']]) == [0]


def test_attention_shape_gradient_and_formula():
    torch = pytest.importorskip('torch')
    m = load('self_attention').SelfAttention(4, 3, 2)
    X = torch.randn(2, 5, 4, requires_grad=True)
    out = m(X)
    expected = torch.softmax(m.Wq(X) @ m.Wk(X).transpose(-1, -2) / 3 ** 0.5, -1) @ m.Wv(X)
    torch.testing.assert_close(out, expected)
    assert out.shape == (2, 5, 2)
    out.sum().backward()
    assert torch.isfinite(X.grad).all()


def test_numpy_attention_matches_matmul_and_single_token():
    rng = np.random.default_rng(7)
    Wq, Wk = rng.normal(size=(2, 4, 3))
    Wv = rng.normal(size=(4, 2))
    model = load('self_attention_numpy').SelfAttentionNumpy(Wq, Wk, Wv)
    X = rng.normal(size=(2, 5, 4))
    scores = (X @ Wq) @ (X @ Wk).transpose(0, 2, 1) / np.sqrt(3)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = (weights / weights.sum(axis=-1, keepdims=True)) @ (X @ Wv)
    np.testing.assert_allclose(model.forward(X), expected, atol=1e-12)
    np.testing.assert_allclose(model.forward(X[:, :1]), X[:, :1] @ Wv)


def test_transformer_blocks_match_without_biases():
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(19)
    E, H = 4, 7
    Wq, Wk, Wv = rng.normal(scale=0.2, size=(3, E, E))
    W1 = rng.normal(scale=0.2, size=(E, H))
    W2 = rng.normal(scale=0.2, size=(H, E))
    X = rng.normal(size=(2, 3, E))
    original = X.copy()
    numpy_block = load('self_attention_numpy').TransformerBlockNumpy(Wq, Wk, Wv, W1, W2)
    torch_block = load('self_attention').TransformerBlock(E, H).double()
    with torch.no_grad():
        for layer, weight in [
            (torch_block.attention.Wq, Wq),
            (torch_block.attention.Wk, Wk),
            (torch_block.attention.Wv, Wv),
            (torch_block.ffn[0], W1),
            (torch_block.ffn[2], W2),
        ]:
            layer.weight.copy_(torch.from_numpy(weight.T))
            layer.bias.zero_()
    tensor = torch.tensor(X, requires_grad=True)
    result = torch_block(tensor)
    np.testing.assert_allclose(numpy_block.forward(X), result.detach().numpy(), atol=1e-10)
    np.testing.assert_array_equal(X, original)
    result.sum().backward()
    assert torch.isfinite(tensor.grad).all()
    assert np.isfinite(numpy_block.forward(np.ones((1, 2, E)))).all()


def test_pooling_padding_negative_values_and_empty_rows():
    pool = load('pooling').masked_pool
    X = np.array([[[-5., -2.], [-3., -4.], [999., 999.]],
                  [[1., 2.], [3., 4.], [5., 6.]]])
    mask = np.array([[1, 1, 0], [0, 0, 0]])
    np.testing.assert_allclose(pool(X, mask), [[-4, -3], [0, 0]])
    np.testing.assert_allclose(pool(X, mask, 'max'), [[-3, -2], [0, 0]])
    np.testing.assert_allclose(pool(X[:, :1], [[1], [1]]), X[:, 0])


def test_logistic_gradient_finite_difference_and_extreme_logits():
    module = load('logistic_regression')
    X = np.array([[1., 2.], [-1., 3.], [0.5, -2.]])
    y, w, b = np.array([1., 0., 1.]), np.array([0.3, -0.2]), 0.1
    _, dw, db = module.loss_and_gradient(X, y, w, b)
    eps = 1e-6
    for j in range(len(w)):
        delta = np.zeros_like(w)
        delta[j] = eps
        numeric = (module.loss_and_gradient(X, y, w + delta, b)[0]
                   - module.loss_and_gradient(X, y, w - delta, b)[0]) / (2 * eps)
        np.testing.assert_allclose(dw[j], numeric, atol=1e-8)
    numeric_b = (module.loss_and_gradient(X, y, w, b + eps)[0]
                 - module.loss_and_gradient(X, y, w, b - eps)[0]) / (2 * eps)
    np.testing.assert_allclose(db, numeric_b, atol=1e-8)
    with np.errstate(over='raise', invalid='raise'):
        assert np.isfinite(module.sigmoid(np.array([-1000., 1000.]))).all()
        loss, _, _ = module.loss_and_gradient(np.array([[-1000.], [1000.]]),
                                              np.array([1., 0.]), np.ones(1), 0.)
        assert np.isfinite(loss)


def test_logistic_learns_and_refits():
    module = load('logistic_regression')
    X, y = np.array([[-2.], [-1.], [1.], [2.]]), np.array([0, 0, 1, 1])
    model = module.LogisticRegressor().fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    assert model.losses[-1] < model.losses[0]
    first = model.w.copy()
    model.fit(X, y)
    np.testing.assert_allclose(model.w, first)


def test_cosine_retrieval_zero_ties_k_and_scale():
    retrieve = load('cosine_retrieval').top_k_cosine
    items = np.array([[0., 1.], [2., 0.], [1., 0.], [0., 0.], [-1., 0.]])
    ids, scores = retrieve([1, 0], items, 99)
    np.testing.assert_array_equal(ids, [1, 2, 0, 3, 4])
    np.testing.assert_allclose(scores, [1, 1, 0, 0, -1])
    np.testing.assert_array_equal(retrieve([0, 0], items, 2)[0], [0, 1])
    assert len(retrieve([1, 0], items, 0)[0]) == 0
    assert len(retrieve([1, 0], np.empty((0, 2)), 3)[0]) == 0
    np.testing.assert_array_equal(retrieve([9, 0], items * 3, 2)[0], [1, 2])


def test_ranking_metrics_auc_ties_and_ndcg():
    metrics = load('ranking_metrics')
    assert metrics.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.9]) == pytest.approx(0.875)
    assert metrics.roc_auc([0, 1, 0, 1], [5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.5)
    assert metrics.roc_auc([1, 0], [0.2, 0.9]) == pytest.approx(0.0)

    sk_metrics = pytest.importorskip('sklearn.metrics')
    rng = np.random.default_rng(123)
    for _ in range(25):
        y = rng.integers(0, 2, size=30)
        if y.sum() in (0, 30):
            y[0], y[1] = 0, 1
        scores = np.round(rng.normal(size=30), 1)
        assert metrics.roc_auc(y, scores) == pytest.approx(sk_metrics.roc_auc_score(y, scores))

    assert metrics.ndcg_at_k([1, 0, 1], [0.9, 0.8, 0.7], k=3) == pytest.approx(
        (1.0 + 1.0 / np.log2(4)) / (1.0 + 1.0 / np.log2(3))
    )
    assert metrics.ndcg_at_k([0, 0, 0], [0.9, 0.5, 0.1], k=2) == 0.0


def test_tabular_gbdt_baseline():
    pytest.importorskip('lightgbm')
    pd = pytest.importorskip('pandas')
    run_baseline = load('tabular_gbdt_baseline').run_tabular_baseline
    rng = np.random.default_rng(9)
    n = 200
    num = rng.normal(size=n)
    cat = np.where(num > 0, 'high', 'low')
    noisy_num = np.where(rng.random(n) < 0.1, np.nan, num)
    y = (num + rng.normal(scale=0.5, size=n) > 0).astype(int)
    df = pd.DataFrame({'signal': noisy_num, 'segment': cat, 'target': y})
    clf = run_baseline(df, target_col='target')
    assert clf is not None
