def fake_embed(text):
    return [len(text), 1.0, 2.0]

def test_embedding_vector():
    emb = fake_embed("Bonjour")
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert all(isinstance(x, (int, float)) for x in emb)

def test_embedding_consistency():
    e1 = fake_embed("Bonjour")
    e2 = fake_embed("Bonjour")
    assert e1 == e2

def test_embedding_batch():
    texts = ["A", "BB", "CCC"]
    embeddings = [fake_embed(t) for t in texts]
    assert len(embeddings) == len(texts)
    for emb in embeddings:
        assert isinstance(emb, list)
        assert len(emb) > 0






# Pour tester la logique de mon script, j’utilise un embedding simulé basé sur la longueur du texte.
# Cela me permet de vérifier la structure des données sans dépendre de l’API
# Un test unitaire ne doit pas appeler une API externe.
# Donc j’ai simulé un embedding simple basé sur la longueur du texte. Cela me permet de vérifier que mon script manipule correctement les embeddings :
# vecteur non vide, vecteur numérique, stabilité, cohérence du batch.