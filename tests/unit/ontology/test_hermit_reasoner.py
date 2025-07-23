import rdflib
from rdflib.namespace import OWL, RDF, RDFS

from deepthought.ontology import OntologyManager


def test_class_hierarchy_inference():
    ex = rdflib.Namespace("http://example.com/")
    onto = OntologyManager()
    onto.add_triples(
        [
            (ex.Person, RDF.type, OWL.Class),
            (ex.Student, RDF.type, OWL.Class),
            (ex.Student, RDFS.subClassOf, ex.Person),
            (ex.joe, RDF.type, ex.Student),
            (ex.joe, RDF.type, OWL.NamedIndividual),
        ]
    )
    facts = onto.infer_facts()
    assert (str(ex.joe), str(RDF.type), str(ex.Person)) in facts
