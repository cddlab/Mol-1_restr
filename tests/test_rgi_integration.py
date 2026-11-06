"""RGI input chemistry and diffusion-loop contracts."""

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import AllChem

from mol.data.inference.json_parser import rdkit_mol_to_atom_array
from mol.model import generator
from rgi_toolkit.combined import CombinedRestraints
from rgi_toolkit.mol1.adapter import Mol1Adapter


@pytest.mark.parametrize("smiles", ["F/C=C/Cl", "c1ccncc1", "C[C@H](O)C(=O)O", "[Na+]"])
def test_smiles_bond_orders_and_charge_survive(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=5) == 0
    aa = rdkit_mol_to_atom_array(mol)
    heavy = Chem.RemoveHs(mol)
    kmol = Chem.Mol(heavy)
    Chem.Kekulize(kmol, clearAromaticFlags=True)
    expected = sorted(
        (
            min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
            max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
            int(b.GetBondTypeAsDouble()),
        )
        for b in kmol.GetBonds()
    )
    assert sorted(map(tuple, aa.bonds.as_array())) == expected
    assert list(aa.charge) == [a.GetFormalCharge() for a in heavy.GetAtoms()]


def restraint(target):
    aa = AtomArray(2)
    aa.set_annotation("label_asym_id", np.array(["A", "B"]))
    aa.set_annotation("mol_type", np.array(["protein", "protein"]))
    aa.res_id = np.array([10, 90])
    aa.res_name = np.array(["ALA", "ALA"])
    aa.atom_name = np.array(["CA", "CA"])
    aa.element = np.array(["C", "C"])
    rgi = CombinedRestraints()
    rgi.setup(
        Mol1Adapter({"atom_array": aa, "atom_to_token_idx": np.arange(2)}),
        config={
            "gpu": False,
            "compile_cpu": False,
            "distance_restraints_config": [
                {
                    "atom_selection1": "chain A",
                    "atom_selection2": "chain B",
                    "harmonic": {"target_distance": target},
                }
            ],
        },
    )
    return rgi


@pytest.mark.parametrize("target", [4.0, 7.0])
def test_diffusion_hook_sigma_step_and_uneven_sample_chunks(monkeypatch, target):
    monkeypatch.setattr(
        generator,
        "centre_random_augmentation",
        lambda x_input_coords, N_sample: x_input_coords.unsqueeze(-3),
    )
    rgi = restraint(target)
    calls = []
    minimize = rgi.minimize

    def record(coords, step, sigma):
        calls.append((coords.shape[0], step, sigma))
        return minimize(coords, step, sigma)

    rgi.minimize = record

    def denoise(**kwargs):
        return (
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            .expand_as(kwargs["x_noisy"])
            .clone()
        )

    with torch.inference_mode():
        coords = generator.sample_diffusion(
            denoise_net=denoise,
            input_feature_dict={"atom_to_token_idx": torch.arange(2)},
            s_inputs=torch.zeros(2, 1),
            s_trunk=None,
            z_trunk=None,
            pair_z=None,
            p_lm=None,
            c_l=None,
            noise_schedule=torch.tensor([1.0, 0.1, 0.0]),
            N_sample=3,
            gamma0=0.8,
            gamma_min=0.0,
            step_scale_eta=1.0,
            diffusion_chunk_size=2,
            combined_restraints=rgi,
        )
    assert [(n, step) for n, step, _ in calls] == [(2, 0), (2, 1), (1, 0), (1, 1)]
    assert [sigma for _, _, sigma in calls] == pytest.approx([1.0, 0.1, 1.0, 0.1])
    assert coords.shape == (3, 2, 3)
    torch.testing.assert_close(
        torch.linalg.norm(coords[:, 0] - coords[:, 1], dim=-1),
        torch.full((3,), target),
        atol=1e-4,
        rtol=0.0,
    )
