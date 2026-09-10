"""tests/test_template_polish.py — Nettoyage du métalangage des gabarits."""
from __future__ import annotations

from ratis_net.skeleton_speaker_v2 import SkeletonSpeakerV2


def test_strip_meta_tail_dense():
    t = ("{X} is defined here as an element of {Y} whose primary role "
         "concerns {Z}, while stating the starting assumptions, within a "
         "scientific frame, in a neutral register.")
    assert SkeletonSpeakerV2._strip_meta_tail(t) == (
        "{X} is defined here as an element of {Y} whose primary role "
        "concerns {Z}.")


def test_strip_meta_tail_conversation_fr():
    t = ("Bonjour {PERSON}. Comment te sens-tu par rapport à {X} "
         "aujourd'hui , tout en respectant le rythme de l'échange, dans un "
         "échange détendu, avec une tonalité calme, avec un registre neutre.")
    assert SkeletonSpeakerV2._strip_meta_tail(t) == (
        "Bonjour {PERSON}. Comment te sens-tu par rapport à {X} aujourd'hui.")


def test_strip_meta_tail_keeps_no_slot_text():
    assert SkeletonSpeakerV2._strip_meta_tail("Hello.") == "Hello."


def test_polish_typography():
    assert SkeletonSpeakerV2._polish("hello , world") == "Hello, world."
    assert SkeletonSpeakerV2._polish("déjà fini !") == "Déjà fini !"
