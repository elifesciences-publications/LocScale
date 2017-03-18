from EMAN2 import *
from sparx import *
import numpy as np
import argparse, os, sys
import subprocess
import math
import gc

basename=os.path.basename(sys.argv[0])
message = """\nThis program is designed to generate a cryo-EM map with optimized local contrast using a reference-based scaling procedure. It requires a unfilteres, unsharpeend 3D reconstruction and an atomic model fitted into this density as input.  \n"""
revision=filter(str.isdigit, "$Revision: 1 $")                            # to be updated by svn after every commit 
datmod  = "q$Date: 2017-03-06 22:14:31 +0200 (Mo, 06 Mar 2017) $"         # to be updated by svn after every commit      
author  ='author: Arjen J. Jakobi, EMBL'+'; '+datmod [8:18]        
version = basename+'  0.1'+'  (r'+revision+';'+datmod [7:18]+')'


print""
cmdl_parser = argparse.ArgumentParser(
               prog=sys.argv[0],
               description='*** Computes real-space correlation of atomic model against map ***',
               formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30), add_help=False)

required_args = cmdl_parser.add_argument_group("Input")
optional_args = cmdl_parser.add_argument_group("Options")

required_args.add_argument('-m1', '--em_map', metavar="em_map.mrc",  type=str, required=True,
                         help='Input filename EM map')
required_args.add_argument('-m2', '--model_map', metavar="model_map.mrc",  type=str, required=True,
                         help='Input filename PDB map')
required_args.add_argument('-dmin', metavar="dmin",  type=float, required=True,
                         help='map resolution')
required_args.add_argument('-p', '--apix', metavar="apix",  type=float, required=True,
                         help='pixel size')
required_args.add_argument('-mask', metavar="mask.mrc", type=str, required=True,
                         help='Input filename mask')
optional_args.add_argument('--wn', type=int, default=None,
                                 help='window size [pix]')
optional_args.add_argument('-h', '--help', action="help",
                         help="show this help message and exit")
optional_args.add_argument('--out',  type=str, default="locscale.mrc",
                         help='Output filename')
optional_args.add_argument('--bfactor', type=float, default=None,
                         help='set bfactor [A^2]')
optional_args.add_argument('--verbose', action="store_true",
                         help='print result to stdout')

args = cmdl_parser.parse_args()

def show(self, log = None):
    if(log is None): log = sys.stdout

map1 = get_image(args.em_map)
map2 = get_image(args.model_map)
dmin = args.dmin
mask = binarize(get_im(args.mask),0.5)
apix = args.apix
if (args.wn == None):
   wn = int(round(9*dmin)) # set default window size to 9 times average resolution
else:
   wn = args.wn 

def which(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def set_zero_origin_and_pixel_size(map,apix):
    """
    """
    #map = EMData()
    #map.read_image(os.path.abspath(args.map.name),0,True)

    map['MRC.nxstart'] = 0
    map['MRC.nystart'] = 0
    map['MRC.nzstart'] = 0
    map.set_attr("apix_x",apix)
    map.set_attr("apix_y",apix)
    map.set_attr("apix_z",apix)
    return map

def hammings_1d(shape):
    from numpy import hamming
    h1d = []
    for a in range(3):
        size = shape[a]
        hamm = hamming(size)
        h1d.append(hamm)
    return h1d

def hannings_1d(shape):
    from numpy import hanning
    h1d = []
    for a in range(3):
        size = shape[a]
        hann = hanning(size)
        h1d.append(hann)
    return h1d

def apply_hann_window(subvolume):
    from numpy import multiply, swapaxes
    h0, h1, h2 = hannings_1d(subvolume.shape)
    multiply(subvolume, h2, subvolume)
    subvolumea1 = swapaxes(subvolume, 1, 2)
    multiply(subvolumea1, h1, subvolumea1)
    subvolumea0 = swapaxes(subvolume, 0, 2)
    multiply(subvolumea0, h0, subvolumea0)

    return subvolume

def compute_radial_amplitude_distribution(map,apix):
    data = map.do_fft()
    radial_average = data.calc_radial_dist(map.get_xsize()/2,0,1.0,0)
    #Util.save_data(0,1.0/(apix*map.get_xsize()),radial_average,'test_data.dat')
    return radial_average

def set_radial_amplitude_distribution(map,amplitude_distribution,apix):
    data = map.do_fft()
    frequency_range = np.arange(0,(1/(2*apix)),(1.0/(apix*map.get_xsize())))
    frequency_range = np.ndarray.tolist(frequency_range[0:len(amplitude_distribution)])
    sf = XYData()
    sf.set_xy_list(frequency_range,amplitude_distribution)
    data.process_inplace("filter.setstrucfac",{"apix":apix,"strucfac":sf})
    
    #print amplitude_distribution
    #data.apply_radial_func(0,(1.0/(apix*map.get_xsize())),amplitude_distribution)
    map = data.do_ift()
    return map

# iterate over each voxel in input map
def run_window_function(map1,map2,mask,wn):
	from EMAN2 import EMNumPy
	import numpy as np
	from sparx import fsc
	map1 = EMNumPy.em2numpy(map1)
	map2 = EMNumPy.em2numpy(map2)
	ni,nj,nk = np.shape(map1)
	#map_new = np.zeros((ni,nj,nk))
	map_new = EMData()
	map_new.set_size(nk,nj,ni)
	map_box1, map_box2 = np.zeros((wn,wn,wn)), np.zeros((wn,wn,wn))
	local_resolution_volume = model_blank(nk,nj,ni)
	# iterate over voxels in map
	c = 0
	for i in xrange(0,30):
		for j in xrange(0,nj):
			for k in xrange(0,nk):
				#print '\r{0}'.format([i,j,k]),
				#c += 1
				map_box1, map_box2 = np.zeros((wn,wn,wn)), np.zeros((wn,wn,wn))
				# iterate over voxels within box to be extracted with search area limited by mask
				if (mask.get_value_at(k,j,i) > 0.5):
					for l,u in zip(xrange(0,wn),xrange(i-int(round(wn/2)),i+int(round(wn/2)+1))):
						for m,v in zip(xrange(0,wn),xrange(j-int(round(wn/2)),j+int(round(wn/2)+1))):
							for n,w in zip(xrange(0,wn),xrange(k-int(round(wn/2)),k+int(round(wn/2)+1))):
								if (u <= (len(map1)-1)):
									p = u	
								elif (u > (len(map1)-1)):
									p = u-len(map1)	
								if (v <= (len(map1)-1)):
									q = v
								elif (v > (len(map1)-1)):
									q = v-len(map1)
								if (w <= (len(map1)-1)):
									r = w
								elif (w > len(map1)):
									r = w-len(map1)
								map_box1[l,m,n] = np.copy(map1[p,q,r])
								map_box2[l,m,n] = np.copy(map2[p,q,r])
						
					# compute radial average and scale input map by reference map
					mb1_radial_average = compute_radial_amplitude_distribution(EMNumPy.numpy2em(map_box1),apix)
					mb2_radial_average = compute_radial_amplitude_distribution(EMNumPy.numpy2em(map_box2),apix)
					map_b_sharpened = set_radial_amplitude_distribution(EMNumPy.numpy2em(map_box1),mb2_radial_average,apix)
					#map_b_sharpened.write_image('b_sharp.mrc')
					#EMNumPy.numpy2em(map_box1).write_image('mb1_before_bsharp.mrc')
					#EMNumPy.numpy2em(map_box2).write_image('mb2_before_bsharp.mrc')
					
					map_new.set_value_at_fast(k,j,i,map_b_sharpened[int(wn/2),int(wn/2),int(wn/2)])
					#print map_b_sharpened[int(wn/2),int(wn/2),int(wn/2)]
					#print map_box1[int(wn/2),int(wn/2),int(wn/2)]	
					
					del map_box1
					del map_box2
					gc.collect()


			#sys.stdout.flush()
	map_final = set_zero_origin_and_pixel_size(map_new,args.apix)
	return map_final

if (__name__ == "__main__"):
	LocSharpVol = run_window_function(map1,map2,mask,wn)
	LocSharpVol.write_image('locsharp_vol01.mrc')
