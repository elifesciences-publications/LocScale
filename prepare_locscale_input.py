"""
Script to generate reference model map for LocScale scaling

Uses cctbx libraries - please cite:
Grosse-Kunstleve RW et al. J. Appl. Cryst. 35:126-136 (2002)

Arjen Jakobi, EMBL (2016) 
"""

import argparse
import os
import sys
import warnings

from cctbx import crystal, maptbx
from iotbx import ccp4_map, file_reader
import iotbx.pdb
from libtbx import group_args
from mmtbx import real_space_correlation
import mmtbx.maps.utils
from scitbx.array_family import flex


warnings.simplefilter('ignore', DeprecationWarning)

progname = os.path.basename(sys.argv[0])
revision = filter(str.isdigit, "$Revision: 1 $")  # to be updated by gitlab after every commit
datmod = "$Date: 2017-03-06 22:14:31 +0200 (Mo, 06 Mar 2017) $"  # to be updated by gitlab fter every commit
author = 'authors: Arjen J. Jakobi and Carsten Sachse, EMBL' + '; ' + datmod [8:18]
version = progname + '  0.1' + '  (r' + revision + ';' + datmod [6:18] + ')'

simple_cmd = 'phenix.python prepare_locscale_input.py -mc model_coordinates.pdb -em em_map.mrc'

cmdl_parser = argparse.ArgumentParser(
description='*** Computes reference map from PDB model and generates files for LocScale ***\n\n' + \
'Example usage: \"{0}\". '.format(simple_cmd) + \
'Generates copies of input files and simulated map ending with 4locscale.xxx, e.g. em_map_4locscale.mrc ' + \
', model_coordinates_4locscale.pdb and model_coordinates_4locscale.mrc. \n' + \
'{0} on {1}'.format(author, datmod))

cmdl_parser.add_argument('-mc', '--model_coordinates', required=True, type=argparse.FileType('r'), help='Input filename PDB model')
cmdl_parser.add_argument('-em', '--em_map', required=True, type=argparse.FileType('r'), help='Input filename EM map')
cmdl_parser.add_argument('-ma', '--mask', type=argparse.FileType('r'), help='Input filename mask')
cmdl_parser.add_argument('-p', '--apix', type=float, help='pixel size in Angstrom')
cmdl_parser.add_argument('-dmin', '--resolution', type=float, help='map resolution in Angstrom')
cmdl_parser.add_argument('-t', '--table', type=str, default="electron", help='Scattering table [electron, itcc]')
cmdl_parser.add_argument('-b', '--b_factor', type=float, default=None, help='set bfactor in [A^2]')
cmdl_parser.add_argument('-o', '--outfile', type=str, default="rscc.dat", help='Output filename for RSCC data')
cmdl_parser.add_argument('--rscc', action="store_true", help='compute per-residue RSCC')


def generate_output_file_names(map, model):
    map_out = os.path.splitext(map.name)[0] + "_4locscale.mrc"
    model_out = os.path.splitext(model.name)[0] + "_4locscale.pdb"
    model_map_out = os.path.splitext(model.name)[0] + "_4locscale.mrc"
    return map_out, model_out, model_map_out

def get_dmin(dmin, target_map):
    if dmin is not None:
        d_min = dmin
        print "Model map will be computed to " + str(d_min) + " Angstrom\n"
    else:
        pixel_size = estimate_pixel_size_from_unit_cell(target_map)
        d_min = round(2 * pixel_size + 0.002, 4)
        print "Model map will be computed to " + str(d_min) + " Angstrom\n"
    return d_min

def set_detail_level_and_radius(d_min):
    if(d_min < 2.0): 
        detail = "atom"
    else:            
        detail = "residue"
    
    if(d_min < 1.0):                    
        atom_radius = 1.0
    elif(d_min >= 1.0 and d_min < 2.0):   
        atom_radius = 1.5
    elif(d_min >= 2.0 and d_min < 4.0): 
        atom_radius = 2.0
    else:                               
        atom_radius = 2.5
    return detail, atom_radius

def check_for_zero_B_factor(xrs):
    xrs = xrs.expand_to_p1(sites_mod_positive=True)
    bs = xrs.extract_u_iso_or_u_equiv()
    sel_zero = bs < 1.e-3
    n_zeros = sel_zero.count(True)
    if (n_zeros > 0):
        print "Input model contains %d atoms with B=0\n" % n_zeros

def print_map_statistics(input_model, target_map):
#     try:
#         assert (not None in [input_model, target_map])
#     except AssertionError:
#         print "Input model or map does not exist. Please provide a valid file."
#         exit(1)
    print "Map dimensions:", target_map.data.all()
    print "Map origin   :", target_map.data.origin()
    print "Map unit cell:", target_map.unit_cell_parameters
    print ""

def get_symmetry_from_target_map(target_map):
    symm = crystal.symmetry(
    space_group_symbol="P1",
    unit_cell=target_map.unit_cell_parameters)
    return symm    

def estimate_pixel_size_from_unit_cell(target_map):
    unit_cell = target_map.unit_cell_parameters
    map_grid = target_map.data.all()
    apix_x = unit_cell[0] / map_grid[0]
    apix_y = unit_cell[1] / map_grid[1]
    apix_z = unit_cell[2] / map_grid[2]
    try: 
        assert (round(apix_x, 4) == round(apix_y, 4) == round(apix_z, 4))
        pixel_size = round(apix_x, 4)
    except AssertionError:
        print "Inconsistent pixel size: %g" % apix_x  
    return pixel_size

def determine_shift_from_map_header(target_map):
    origin = target_map.data.origin()
    translation_vector = [0 - target_map.data.origin()[0], 0 - target_map.data.origin()[1], 0 - target_map.data.origin()[2]]
    return translation_vector 

def shift_map_to_zero_origin(target_map, cg, map_out):
    em_data = target_map.data.as_double()
    em_data = em_data.shift_origin()
    ccp4_map.write_ccp4_map(
        file_name=map_out,
        unit_cell=cg.unit_cell(),
        space_group=cg.space_group(),
        map_data=em_data,
        labels=flex.std_string([""]))
    shifted_map = file_reader.any_file(map_out).file_object
    return em_data, shifted_map

def apply_shift_transformation_to_model(input_model, target_map, symm, pixel_size, model_out):
    sg = symm.space_group()
    uc = symm.unit_cell()
    pdb_hierarchy = input_model.construct_hierarchy().deep_copy()
    atoms = pdb_hierarchy.atoms()
    sites_frac = uc.fractionalize(sites_cart=atoms.extract_xyz())
    if pixel_size is None:
        pixel_size = estimate_pixel_size_from_unit_cell(target_map)
    translation_vector = determine_shift_from_map_header(target_map)
    translation_vector[0], translation_vector[1], translation_vector[2] = (translation_vector[0] * pixel_size) / uc.parameters()[0], (translation_vector[1] * pixel_size) / uc.parameters()[1], (translation_vector[2] * pixel_size) / uc.parameters()[2]
    new_sites = sites_frac + translation_vector
    translation_vector[0], translation_vector[1], translation_vector[2] = translation_vector[0] * uc.parameters()[0], translation_vector[1] * uc.parameters()[1], translation_vector[2] * uc.parameters()[2]
    atoms.set_xyz(uc.orthogonalize(sites_frac=new_sites))
    f = open(model_out, "w")
    f.write(pdb_hierarchy.as_pdb_string(crystal_symmetry=symm))
    f.close()

def set_isotropic_b_factor(xrs,b_factor):
    xrs.convert_to_isotropic
    xrs = xrs.set_b_iso(value=b_factor)
    return xrs 

def compute_model_map(xrs, target_map, symm, d_min, table, model_map_out):
    xrs.scattering_type_registry(
    d_min=d_min,
    table=table)
    fc = xrs.structure_factors(d_min=d_min).f_calc()
    cg = maptbx.crystal_gridding(
    unit_cell=symm.unit_cell(),
    space_group_info=symm.space_group_info(),
    pre_determined_n_real=target_map.data.all())
    fc_map = fc.fft_map(
    crystal_gridding=cg).apply_sigma_scaling().real_map_unpadded()
    try:
        assert (fc_map.all() == fc_map.focus() == target_map.data.all())
    except AssertionError:
        print "Different dimension of experimental and simulated model map."
    ccp4_map.write_ccp4_map(
        file_name=model_map_out,
        unit_cell=cg.unit_cell(),
        space_group=cg.space_group(),
        map_data=fc_map,
        labels=flex.std_string([""]))
    return cg, fc_map 

def compute_real_space_correlation_simple(fc_map, em_data):
    cc_overall_cell = flex.linear_correlation(x=em_data.as_1d(),
                      y=fc_map.as_1d()).coefficient()
    print "\nOverall real-space correlation (unit cell)   : %g\n" % cc_overall_cell

def compute_real_space_correlation(xrs, input_model, fc_map, shifted_map, em_data, cg, symm, rscc_out, detail, atom_radius):
    unit_cell_for_interpolation = shifted_map.grid_unit_cell()
    frac_matrix = unit_cell_for_interpolation.fractionalization_matrix()
    sites_cart = xrs.sites_cart()
    sites_frac = xrs.sites_frac()
    pdb_hierarchy = input_model.construct_hierarchy()
    pdb_hierarchy.atoms().reset_i_seq()
    results = []
    atom_radius = atom_radius
    # need to change these lines - some problems with hyrogens
    hydrogen_atom_radius = 0.8
    use_hydrogens = False
    unit_cell = cg.unit_cell()
    for chain in pdb_hierarchy.chains():
      for residue_group in chain.residue_groups():
        for conformer in residue_group.conformers():
          for residue in conformer.residues():
            residue_id_str = "%2s %1s %3s %4s %1s" % (chain.id, conformer.altloc,
                                                      residue.resname, residue.resseq, residue.icode)
            residue_sites_cart = flex.vec3_double()
            residue_b = flex.double()
            residue_occ = flex.double()
            residue_mv1 = flex.double()
            residue_mv2 = flex.double()
            residue_rad = flex.double()
            for atom in residue.atoms():
                atom_id_str = "%s %4s" % (residue_id_str, atom.name)
                if (atom.element_is_hydrogen()): 
                    rad = hydrogen_atom_radius
                else: 
                    rad = atom_radius

                if (not (atom.element_is_hydrogen() and not use_hydrogens)):
                    map_value_em = em_data.eight_point_interpolation(unit_cell.fractionalize(atom.xyz))
                    map_value_fc = fc_map.eight_point_interpolation(unit_cell.fractionalize(atom.xyz))
                    residue_sites_cart.append(atom.xyz)
                    residue_b.append(atom.b)
                    residue_occ.append(atom.occ)
                    residue_mv1.append(map_value_em)
                    residue_mv2.append(map_value_fc)
                    residue_rad.append(rad)
            if (detail == "residue"):
                sel = maptbx.grid_indices_around_sites(
                unit_cell=unit_cell,
                fft_n_real=shifted_map.data.focus(),
                fft_m_real=shifted_map.data.all(),
                sites_cart=residue_sites_cart,
                site_radii=residue_rad)
                cc = flex.linear_correlation(x=em_data.select(sel),
                y=fc_map.select(sel)).coefficient()
                result = group_args(
                residue=residue,
                chain_id=chain.id,
                id_str=residue_id_str,
                cc=cc,
                map_value_em=flex.mean(residue_mv1),
                map_value_fc=flex.mean(residue_mv2),
                b=flex.mean(residue_b),
                occupancy=flex.mean(residue_occ),
                n_atoms=residue_sites_cart.size())
                results.append(result)
    f = open(rscc_out, 'w')
    cc_sum = 0
    for result in results:
        cc_sum += result.cc
    cc_overall_model = cc_sum / len(results)
    cc_overall_cell = flex.linear_correlation(x=em_data.as_1d(), y=fc_map.as_1d()).coefficient()

    print "\nOverall real-space correlation (around atoms): %g" % cc_overall_model
    print "Overall real-space correlation (unit cell)   : %g\n" % cc_overall_cell
    for result in results:
        f.write(result.id_str + "  " + str(result.cc) + "\n")
    f.close()

def prepare_reference_and_experimental_map_for_locscale (args, out=sys.stdout):
    """
    """  
    map_out, model_out, model_map_out = generate_output_file_names(args.em_map, args.model_coordinates)
    target_map = file_reader.any_file(args.em_map.name).file_object
    input_model = file_reader.any_file(args.model_coordinates.name).file_object
    d_min = get_dmin(args.resolution, target_map)
    sc_table = args.table
  
    print_map_statistics(input_model, target_map)
    symm = get_symmetry_from_target_map(target_map)
    apply_shift_transformation_to_model(input_model, target_map, symm, args.apix, model_out)
    shifted_model = iotbx.pdb.hierarchy.input(file_name=model_out)
    xrs = shifted_model.xray_structure_simple(crystal_symmetry=symm)
    if args.b_factor is not None:
       xrs = set_isotropic_b_factor(xrs,args.b_factor)
    check_for_zero_B_factor(xrs) 
    cg, fc_map = compute_model_map(xrs, target_map, symm, d_min, sc_table, model_map_out)
    
    try: 
        assert (fc_map.all() == fc_map.focus() == target_map.data.all())
    except AssertionError:
        print "Different dimension of experimental and simulated model map."
  
    em_data, shifted_map = shift_map_to_zero_origin(target_map, cg, map_out)
    
    if args.rscc:
        detail, atom_radius = set_detail_level_and_radius(d_min)
        compute_real_space_correlation(xrs, shifted_model, fc_map, shifted_map, em_data, cg, symm, args.outfile, detail, atom_radius)
    else:
        compute_real_space_correlation_simple(em_data, fc_map)	

if (__name__ == "__main__") :
    args = cmdl_parser.parse_args()
    prepare_reference_and_experimental_map_for_locscale(args)
