from __future__ import annotations

from io import BytesIO
from typing import Iterable, Tuple

from owlready2 import ThingClass, World, sync_reasoner_hermit
from rdflib import Graph, URIRef
from rdflib.namespace import RDF

Triple = Tuple[str, str, str]


class OntologyManager:
    """Manage an OWL/RDF ontology and run reasoning using HermiT."""

    def __init__(self) -> None:
        self.graph = Graph()

    def add_triple(
        self, subject: str | URIRef, predicate: str | URIRef, obj: str | URIRef
    ) -> None:
        """Add a triple to the underlying graph."""
        self.graph.add((URIRef(str(subject)), URIRef(str(predicate)), URIRef(str(obj))))

    def add_triples(self, triples: Iterable[Triple]) -> None:
        for s, p, o in triples:
            self.add_triple(s, p, o)

    def infer_facts(self) -> list[Triple]:
        """Run the HermiT reasoner and return inferred triples."""
        data = self.graph.serialize(format="xml").encode("utf-8")
        world = World()
        onto = world.get_ontology("http://deepthought.local/onto.owl").load(
            fileobj=BytesIO(data)
        )
        sync_reasoner_hermit(world)
        facts: list[Triple] = []
        for ind in onto.individuals():
            for cls in ind.INDIRECT_is_a:
                if isinstance(cls, ThingClass):
                    facts.append((ind.iri, str(RDF.type), cls.iri))
        return facts
