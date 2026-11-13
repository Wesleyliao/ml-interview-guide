"""Behavioral regressions for the supplied AI live-coding reference answers."""
import asyncio
import copy
import itertools
import math
import random
import pytest
from .solutions import chunk_tokens, answer_query, rank_documents, run_agent, evaluate_runs


def test_chunk_boundaries_exhaustive():
    for n,size in itertools.product(range(12), range(1,8)):
        for overlap in range(size):
            tokens = list(range(n))
            result = chunk_tokens(tokens,size,overlap)
            if not n:
                assert result == []
                continue
            assert result[0][0] == 0
            assert result[-1][0]+len(result[-1][1]) == n
            for index,(start,chunk) in enumerate(result):
                assert chunk == tokens[start:start+size]
                assert 0 < len(chunk) <= size
                if index:
                    assert start-result[index-1][0] == size-overlap
                if index < len(result)-1:
                    assert start+size < n
            assert set(x for _,chunk in result for x in chunk) == set(tokens)
            result[0][1][0] = 'mutated'
            assert tokens == list(range(n))


@pytest.mark.parametrize('size,overlap',[(0,0),(-1,0),(3,-1),(3,3),(3,4)])
def test_chunk_invalid_even_if_empty(size,overlap):
    with pytest.raises(ValueError):
        chunk_tokens([],size,overlap)


DOCS = [
    {'id':'a','text':'public','readers':{'u'},'external_ok':True},
    {'id':'b','text':'internal','readers':{'u'},'external_ok':False},
    {'id':'c','text':'secret','readers':{'v'},'external_ok':True}]


def test_rag_fallback_scope_and_no_mutation():
    async def scenario():
        original = copy.deepcopy(DOCS)
        seen = []
        async def fetch(q):
            seen.append(('fetch',q))
            return original
        async def primary(q,docs):
            seen.append(('primary',[d['id'] for d in docs]))
            raise TimeoutError
        async def fallback(q,docs):
            seen.append(('fallback',[d['id'] for d in docs]))
            return {'text':'ok','citations':['a']}
        result = await answer_query('q','u',fetch,primary,fallback)
        assert result == {'status':'answered','text':'ok','citations':['a']}
        assert seen == [('fetch','q'),('primary',['a','b']),('fallback',['a'])]
        assert original == DOCS
    asyncio.run(scenario())


@pytest.mark.parametrize('error',[RuntimeError('bug'),asyncio.CancelledError()])
def test_rag_does_not_swallow_primary_errors(error):
    async def scenario():
        async def fetch(q): return DOCS
        async def primary(q,docs): raise error
        async def fallback(q,docs): raise AssertionError('Unexpected fallback')
        with pytest.raises(type(error)):
            await answer_query('q','u',fetch,primary,fallback)
    asyncio.run(scenario())


@pytest.mark.parametrize('stage',['fetch','fallback'])
def test_rag_timeout_outside_primary_propagates(stage):
    async def scenario():
        async def fetch(q):
            if stage == 'fetch': raise TimeoutError('fetch')
            return DOCS
        async def primary(q,docs): raise TimeoutError('primary')
        async def fallback(q,docs): raise TimeoutError('fallback')
        with pytest.raises(TimeoutError,match=stage):
            await answer_query('q','u',fetch,primary,fallback)
    asyncio.run(scenario())


@pytest.mark.parametrize('text,citations',[(' ',['a']),('ok',[]),('ok',['a','b']),('ok',['c'])])
def test_rag_validates_actual_fallback_context(text,citations):
    async def scenario():
        async def fetch(q): return DOCS
        async def primary(q,docs): raise TimeoutError
        async def fallback(q,docs): return {'text':text,'citations':citations}
        assert await answer_query('q','u',fetch,primary,fallback) == {'status':'abstained'}
    asyncio.run(scenario())


def test_rag_empty_contexts_skip_model_calls():
    async def scenario():
        async def forbidden(*args): raise AssertionError('must not call')
        async def fetch(q): return DOCS
        assert await answer_query('q','nobody',fetch,forbidden,forbidden) == {'status':'abstained'}
        async def internal_only(q): return [DOCS[1]]
        async def primary(q,docs): raise TimeoutError
        assert await answer_query('q','u',internal_only,primary,forbidden) == {'status':'abstained'}
    asyncio.run(scenario())


def test_retrieval_cosine_not_dot_product_and_scale_invariance():
    docs = [{'id':i,'vector':v,'readers':{'u'}} for i,v in
            [('large',[100,100]),('aligned',[1,0]),('negative',[-1,0]),('zero',[0,0])]]
    docs.insert(0,{'id':'hidden','vector':[1,0],'readers':{'other'}})
    saved = copy.deepcopy(docs)
    assert rank_documents([1,0],docs,'u',3) == ['aligned','large','negative']
    for doc in docs:
        doc['vector'] = [x*7 for x in doc['vector']]
    assert rank_documents([5,0],docs,'u',3) == ['aligned','large','negative']
    rank_documents([1,0],saved,'u',2)
    assert saved[1]['vector'] == [100,100]


def test_retrieval_exact_ties_independent_of_input_order():
    docs = [{'id':i,'vector':[0,-1],'readers':{'u'}} for i in ['c','a','b']]
    for ordering in itertools.permutations(docs):
        assert rank_documents([1,0],list(ordering),'u',2) == ['a','b']


def test_agent_transcript_and_final_after_budget():
    seen = []
    def model(messages):
        seen.append(copy.deepcopy(messages))
        if len(seen) == 1:
            return {'type':'tool','name':'lookup','arguments':{'x':3}}
        return {'type':'final','text':'answer'}
    assert run_agent('q',model,{'lookup':lambda x:str(x*2)},1) == {'status':'answered','text':'answer'}
    assert seen[0] == [{'role':'user','content':'q'}]
    assert seen[1] == [
        {'role':'user','content':'q'},
        {'role':'assistant','content':{'type':'tool','name':'lookup','arguments':{'x':3}}},
        {'role':'tool','name':'lookup','content':'6'}]


@pytest.mark.parametrize('budget',[0,1,4])
def test_agent_exact_model_and_tool_budget(budget):
    models,tools = [],[]
    def model(messages):
        models.append(copy.deepcopy(messages))
        return {'type':'tool','name':'read','arguments':{}}
    def tool():
        tools.append(1)
        return 'value'
    assert run_agent('q',model,{'read':tool},budget) == {'status':'limit','text':None}
    assert len(models) == budget+1 and len(tools) == budget
    assert len(models[-1]) == 1+2*budget


@pytest.mark.parametrize('response',[
    None, [], {}, {'type':'other'}, {'type':'final','text':''},
    {'type':'final','text':7}, {'type':'final','text':'x','extra':1},
    {'type':'tool','name':'unknown','arguments':{}},
    {'type':'tool','name':[],'arguments':{}},
    {'type':'tool','name':'read','arguments':[]},
    {'type':'tool','name':'read','arguments':{1:'bad'}},
    {'type':'tool','name':'read'},
])
def test_agent_validation_precedes_execution_and_budget(response):
    def forbidden(**kwargs): raise AssertionError('must not execute')
    with pytest.raises(ValueError):
        run_agent('q',lambda m:response,{'read':forbidden},0)


def test_agent_exception_and_negative_budget():
    def broken(): raise RuntimeError('tool failed')
    request = {'type':'tool','name':'read','arguments':{}}
    with pytest.raises(RuntimeError,match='tool failed'):
        run_agent('q',lambda m:request,{'read':broken},1)
    with pytest.raises(ValueError):
        run_agent('q',lambda m:None,{},-1)


def test_evaluation_against_materialized_oracle():
    rng = random.Random(72)
    for _ in range(100):
        gold = {str(i):{x for x in 'abcd' if rng.randrange(2)} for i in range(5)}
        rows = [{'query_id':q,'ranked_ids':[rng.choice('abcdxyz') for _ in range(rng.randrange(10))]}
                for q in gold if rng.randrange(2)]
        k = rng.randrange(1,6)
        rankings = {r['query_id']:list(dict.fromkeys(r['ranked_ids']))[:k] for r in rows}
        precisions,recalls,rrs = [],[],[]
        for q,relevant in gold.items():
            if not relevant: continue
            ranking = rankings.get(q,[])
            positions = [i+1 for i,x in enumerate(ranking) if x in relevant]
            precisions.append(len(positions)/k)
            recalls.append(len(positions)/len(relevant))
            rrs.append(1/min(positions) if positions else 0)
        result = evaluate_runs(gold,iter(rows),k)
        assert result['queries'] == len(precisions)
        for name,values in [('precision',precisions),('recall',recalls),('mrr',rrs)]:
            assert math.isclose(result[name],sum(values)/len(values) if values else 0)


def test_evaluation_denominators_and_validation():
    assert evaluate_runs({'a':{'x'},'b':{'y'},'skip':set()},
                         [{'query_id':'a','ranked_ids':['x','x']}],4) == {
                             'queries':2,'precision':0.125,'recall':0.5,'mrr':0.5}
    with pytest.raises(ValueError): evaluate_runs({},[],0)
    with pytest.raises(ValueError): evaluate_runs({},[{'query_id':'bad','ranked_ids':[]}],1)
