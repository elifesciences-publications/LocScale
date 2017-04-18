from EMAN2 import EMData, EMNumPy, Util, XYData
import numpy as np
from sparx import get_image, binarize, filt_gaussl, model_square
import argparse, os, sys

progname = os.path.basename(sys.argv[0])
#message = "\nThis program is designed to generate a cryo-EM map with optimized local contrast using a " + \
#"reference-based scaling procedure. It requires a unfiltered, unsharpened 3D reconstruction and an atomic " + \
#"model fitted into this density as input.  \n"
revision = filter(str.isdigit, "$Revision: 1 $")  # to be updated by gitlab after every commit 
datmod = "$Date: 2017-03-06 22:14:31 +0200 (Mo, 06 Mar 2017) $"  # to be updated by gitlab fter every commit      
author = 'authors: Arjen J. Jakobi and Carsten Sachse, EMBL' + '; ' + datmod [8:18]        
version = progname + '  0.1' + '  (r' + revision + ';' + datmod [7:18] + ')'
     
simple_cmd = 'python locscale.py -em map1.mrc -mm map2.mrc -ma mask.mrc -p 1.0 -w 10 -o scaled.mrc'

cmdl_parser = argparse.ArgumentParser(
description='*** Computes contrast-enhanced cryo-EM maps by local amplitude scaling using a reference model ***' + \
'Example usage: \"{0}\". {1} on {2}'.format(simple_cmd, author, datmod))
     
mpi_cmd = 'mpirun -np 4 python locscale.py -em map1.mrc -mm map2.mrc -ma mask.mrc -p 1.0 -w 10 -mpi -o scaled.mrc'

cmdl_parser.add_argument('-em', '--em_map', required=True, help='Input filename EM map')
cmdl_parser.add_argument('-mm', '--model_map', required=True, help='Input filename PDB map')
cmdl_parser.add_argument('-p', '--apix', type=float, required=True, help='pixel size in Angstrom')
cmdl_parser.add_argument('-ma', '--mask', help='Input filename mask')
cmdl_parser.add_argument('-w', '--window_size', type=int, help='window size in pixel')
cmdl_parser.add_argument('-o', '--outfile', required=True, help='Output filename')
cmdl_parser.add_argument('-mpi', '--mpi', action='store_true', default=False,
                         help='MPI version call by: \"{0}\"'.format(mpi_cmd))
     
def setup_test_data(voldim=30, size=10):
    from sparx import model_gauss
    map1 = model_gauss(size, voldim, voldim, voldim)
    map2 = EMData()
    map2.set_size(voldim, voldim, voldim)
    map2.process_inplace("testimage.noise.gauss", {"sigma":1, "seed":99})
    mask = model_square(size, voldim, voldim, voldim)
    
    return map1, map2, mask

def setup_test_data_to_files(map1_name='map1.mrc', map2_name='map2.mrc', mask_name='mask.mrc'):
    """
    >>> map1_name, map2_name, mask_name = setup_test_data_to_files()
    >>> import subprocess
    >>> n = subprocess.call(simple_cmd.split())
    >>> scaled_vol = get_image('scaled.mrc')
    >>> np.copy(EMNumPy.em2numpy(scaled_vol))[scaled_vol.get_xsize() / 2][scaled_vol.get_ysize() / 2]
    array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
           -1.34984803, -1.43491805, -1.48072839, -1.51072693, -1.59607637,
           -1.63734567, -1.62106037, -1.57305634, -1.52845633, -1.46542633,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ], dtype=float32)
    >>> n = [os.remove(each_file) for each_file in [map1_name, map2_name, mask_name, 'scaled.mrc']]
    """
    map1, map2, mask = setup_test_data()

    map1.write_image(map1_name)
    map2.write_image(map2_name)
    mask.write_image(mask_name)
    
    return map1_name, map2_name, mask_name

def compute_radial_amplitude_distribution(map, apix):
    data = map.do_fft()
    radial_average = data.calc_radial_dist(map.get_xsize() / 2, 0, 1.0, 0)
    # Util.save_data(0,1.0/(apix*map.get_xsize()),radial_average,'test_data.dat')
    return radial_average
  
def set_zero_origin_and_pixel_size(map, apix):
    map['MRC.nxstart'] = 0
    map['MRC.nystart'] = 0
    map['MRC.nzstart'] = 0
    map.set_attr("apix_x", apix)
    map.set_attr("apix_y", apix)
    map.set_attr("apix_z", apix)
    return map
  
def set_radial_amplitude_distribution(map, amplitude_distribution, apix):
    data = map.do_fft()
    frequency_range = np.arange(0, (1 / (2 * apix)), (1.0 / (apix * map.get_xsize())))
    frequency_range = np.ndarray.tolist(frequency_range[0:len(amplitude_distribution)])
    sf = XYData()
    sf.set_xy_list(frequency_range, amplitude_distribution)
    data.process_inplace("filter.setstrucfac", {"apix":apix, "strucfac":sf})
      
    map = data.do_ift()
    return map
  
def get_xyz_locs_and_indices_after_edge_cropping_and_masking(mask, wn):
    mask = np.copy(EMNumPy.em2numpy(mask))
    nk, nj, ni = mask.shape

    kk, jj, ii = np.meshgrid(np.arange(wn / 2, nk - wn / 2), np.arange(wn / 2, nj - wn / 2), np.arange(wn / 2 , ni - wn / 2))
    xzy_locs = np.array([each for each in zip(kk.ravel(), jj.ravel(), ii.ravel())])
    
    cropped_3d_mask_arr = np.array(mask, dtype=bool)[wn / 2:nk - wn / 2, wn / 2:nj - wn / 2, wn / 2:ni - wn / 2 ]
    mask_lin_arr = cropped_3d_mask_arr.ravel()
    
    masked_xyz_locs = xzy_locs[mask_lin_arr]
    indices = np.arange(mask_lin_arr.size)
    masked_indices = indices[mask_lin_arr]
    
    return masked_xyz_locs, masked_indices, cropped_3d_mask_arr.shape, mask.shape

def excise_window_and_mask(map, k, j, i, wn, pd, mask):
    map_box = map[k - wn / 2:k + wn / 2, j - wn / 2: j + wn / 2, i - wn / 2: i + wn / 2]
    map_box = EMNumPy.numpy2em(np.copy(map_box))
    if pd > 1:
        pd_size = int(pd * wn)
        map_box = Util.pad(map_box, pd_size, pd_size, pd_size, 0, 0, 0, 'average')
    
    map_box *= mask

    return map_box

def get_central_scaled_pixel_vals_after_scaling(map1, map2, masked_xyz_locs, wn, apix, pd):
    sharpened_vals = np.array([])
    map1 = np.copy(EMNumPy.em2numpy(map1))
    map2 = np.copy(EMNumPy.em2numpy(map2))
    
    falloff_width = 0.1 * pd * wn / 2.0
    square_dim = int(2 * falloff_width + pd * wn / 2.0)
    mask = model_square(square_dim, int(pd * wn), int(pd * wn), int(pd * wn))
    mask = filt_gaussl(mask, 0.44 / falloff_width)
    for k, j, i in masked_xyz_locs:
        map_box1 = excise_window_and_mask(map1, k, j, i, wn, pd, mask)
        map_box2 = excise_window_and_mask(map2, k, j, i, wn, pd, mask)
        
        # compute radial average and scale input map by reference map
        mb2_radial_average = compute_radial_amplitude_distribution(map_box2, apix)
        map_b_sharpened = set_radial_amplitude_distribution(map_box1, mb2_radial_average, apix)
          
        central_pix = int(round(pd * wn / 2.0))
        sharpened_vals = np.append(sharpened_vals, map_b_sharpened[central_pix, central_pix, central_pix])
          
    return sharpened_vals

def put_scaled_voxels_back_in_original_volume_including_padding(sharpened_vals, masked_indices, cropped_3d_mask_shape,
map_shape):
    map_scaled = np.zeros(np.prod(cropped_3d_mask_shape))
    map_scaled[masked_indices] = sharpened_vals
    z, y, x = cropped_3d_mask_shape
    map_scaled = map_scaled.reshape((y, z, x))
    map_scaled = np.swapaxes(map_scaled, 0, 1)
    
    map_scaled = EMNumPy.numpy2em(np.copy(map_scaled))
    map_scaled = Util.pad(map_scaled, map_shape[2], map_shape[1], map_shape[0], 0, 0, 0, '0')
    
    return map_scaled

def run_window_function_including_scaling(map1, map2, mask, wn, apix, pd=1.0):
    """
    >>> map1, map2, mask = setup_test_data()
    >>> scaled_vol = run_window_function_including_scaling(map1,map2,mask,wn=10,apix=1.0)
    >>> np.copy(EMNumPy.em2numpy(scaled_vol))[scaled_vol.get_xsize() / 2][scaled_vol.get_ysize() / 2]
    array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            1.05564773,  1.02000415,  1.0409112 ,  1.10609078,  1.10587108,
            1.00325978,  0.94142663,  0.9879443 ,  1.00159299,  1.03976917,
            0.95093048,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ], dtype=float32)
    """
    masked_xyz_locs, masked_indices, cropped_3d_mask_shape, map_shape = \
    get_xyz_locs_and_indices_after_edge_cropping_and_masking(mask, wn)
 
    sharpened_vals = get_central_scaled_pixel_vals_after_scaling(map1, map2, masked_xyz_locs, wn, apix, pd)
     
    map_scaled = put_scaled_voxels_back_in_original_volume_including_padding(sharpened_vals, masked_indices,
    cropped_3d_mask_shape, map_shape)
 
    return map_scaled
   
def split_sequence_evenly(seq, size):
    """
    >>> split_sequence_evenly(list(range(9)), 4)
    [[0, 1], [2, 3, 4], [5, 6], [7, 8]]
    >>> split_sequence_evenly(list(range(18)), 4)
    [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13], [14, 15, 16, 17]]
    """
    newseq = []
    splitsize = 1.0 / size * len(seq)
    for i in range(size):
        newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
    return newseq
    
def merge_sequence_of_sequences(seq):
    """
    >>> merge_sequence_of_sequences([list(range(9)), list(range(3))])
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2]
    >>> merge_sequence_of_sequences([list(range(9)), [], list(range(3))])
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2]
    """
    newseq = [number for sequence in seq for number in sequence]
    
    return newseq
    
def compute_padding_average(map1,mask):
    volume_stats_outside_mask = Util.infomask(map1, mask, False)
    average_padding_intensity = volume_stats_outside_mask[0]
    
    return average_padding_intensity
    
def check_for_window_bleeding(mask,wn):
    masked_xyz_locs, masked_indices, cropped_3d_mask_shape, mask_shape = \
    get_xyz_locs_and_indices_after_edge_cropping_and_masking(mask, 0)
    
    zs, ys, xs = zip(*masked_xyz_locs)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    nk, nj, ni = mask_shape

    if xs.min() < wn / 2 or xs.max() > (ni - wn / 2) or \
    ys.min() < wn / 2 or ys.max() > (nj - wn / 2) or \
    zs.min() < wn / 2 or zs.max() > (nk - wn / 2):
        window_bleed = True
    else:
        window_bleed = False
    
    return window_bleed 	
    
def prepare_mask_and_maps_for_scaling(args, map1, get_image, map2):
    if args.mask is None:
        mask = EMData()
        xsize, ysize, zsize = map1.get_xsize(), map1.get_ysize(), map1.get_zsize()
        mask.set_size(xsize, ysize, zsize)
        mask.to_zero()
        if xsize == ysize and xsize == zsize and ysize == zsize:
            sphere_radius = xsize // 2
            mask.process_inplace("testimage.circlesphere", {"radius":sphere_radius})
        else:
            mask += 1
            mask = Util.window(mask, xsize - 1, ysize - 1, zsize -1)
            mask = Util.pad(mask, xsize, ysize, zsize, 0, 0, 0, '0')
    elif args.mask is not None:
        mask = binarize(get_image(args.mask), 0.5)
        
    padded_size = 2
    if args.window_size is None:
        wn = padded_size * int(round(7 * 3 * apix)) # set default window size to 7 times average resolution
    elif args.window_size is not None:
        wn = padded_size * args.window_size

    window_bleed_and_pad = check_for_window_bleeding(mask, wn)
    if window_bleed_and_pad:
        pad_int_map1 = compute_padding_average(map1, mask)
        pad_int_map2 = compute_padding_average(map2, mask)
        map_shape = [(map1.get_xsize() + wn / 2), (map1.get_ysize() + wn / 2), (map1.get_zsize() + wn / 2)] # maybe too large; could pad just to original size + wn/2
        map1 = Util.pad(map1, map_shape[0], map_shape[1], map_shape[2], 0, 0, 0, 'pad_int_map1')
        map2 = Util.pad(map2, map_shape[0], map_shape[1], map_shape[2], 0, 0, 0, 'pad_int_map2')
        mask = Util.pad(mask, map_shape[0], map_shape[1], map_shape[2], 0, 0, 0, '0')
        
    return map1, map2, mask, wn, window_bleed_and_pad

def run_window_function_including_scaling_mpi(map1, map2, mask, wn, apix, pd=1.0):
    """
    >>> map1_name, map2_name, mask_name = setup_test_data_to_files()
    >>> import subprocess
    >>> n = subprocess.call(mpi_cmd.split())
    >>> scaled_vol = get_image('scaled.mrc')
    >>> np.copy(EMNumPy.em2numpy(scaled_vol))[scaled_vol.get_xsize() / 2][scaled_vol.get_ysize() / 2]
    array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
           -1.34984803, -1.43491805, -1.48072839, -1.51072693, -1.59607637,
           -1.63734567, -1.62106037, -1.57305634, -1.52845633, -1.46542633,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        ,  0.        ,  0.        ], dtype=float32)
    >>> n = [os.remove(each_file) for each_file in [map1_name, map2_name, mask_name, 'scaled.mrc']]
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
         
    if rank == 0:
        masked_xyz_locs, masked_indices, cropped_3d_mask_shape, map_shape = \
        get_xyz_locs_and_indices_after_edge_cropping_and_masking(mask, wn)
         
        zs, ys, xs = zip(*masked_xyz_locs)
        zs = split_sequence_evenly(zs, size)
        ys = split_sequence_evenly(ys, size)
        xs = split_sequence_evenly(xs, size)
    else:
        zs = None
        ys = None
        xs = None
     
    zs = comm.scatter(zs, root=0)
    ys = comm.scatter(ys, root=0)
    xs = comm.scatter(xs, root=0)
 
    masked_xyz_locs = zip(zs, ys, xs)
 
    sharpened_vals = get_central_scaled_pixel_vals_after_scaling(map1, map2, masked_xyz_locs, wn, apix, pd)
    sharpened_vals = comm.gather(sharpened_vals, root=0)
     
    if rank == 0:
        sharpened_vals = merge_sequence_of_sequences(sharpened_vals)
 
        map_scaled = put_scaled_voxels_back_in_original_volume_including_padding(np.array(sharpened_vals), masked_indices,
        cropped_3d_mask_shape, map_shape)
    else:
        map_scaled = None
     
    comm.barrier()
 
    return map_scaled, rank
  
def write_out_final_volume_window_back_if_required(args, wn, window_bleed_and_pad, LocScaleVol):
    LocScaleVol = set_zero_origin_and_pixel_size(LocScaleVol, args.apix)
    if window_bleed_and_pad:
        map_shape = [(LocScaleVol.get_xsize() - wn / 2), (LocScaleVol.get_ysize() - wn / 2), (LocScaleVol.get_zsize() - wn / 2)]
        LocScaleVol = Util.window(LocScaleVol, map_shape[0], map_shape[1], map_shape[2])
    LocScaleVol.write_image(args.outfile)

    return LocScaleVol

def launch_amplitude_scaling(args):
    map1 = get_image(args.em_map)
    map2 = get_image(args.model_map)

    map1, map2, mask, wn, window_bleed_and_pad = prepare_mask_and_maps_for_scaling(args, map1, get_image, map2)
     
    if not args.mpi:
        LocScaleVol = run_window_function_including_scaling(map1, map2, mask, wn, args.apix)
        LocScaleVol = write_out_final_volume_window_back_if_required(args, wn, window_bleed_and_pad, LocScaleVol)
    elif args.mpi:
        LocScaleVol, rank = run_window_function_including_scaling_mpi(map1, map2, mask, wn, args.apix)
        if rank == 0:
            LocScaleVol = write_out_final_volume_window_back_if_required(args, wn, window_bleed_and_pad, LocScaleVol)

def main():
    args = cmdl_parser.parse_args()
    
    launch_amplitude_scaling(args)
  
if __name__ == '__main__':
    main()