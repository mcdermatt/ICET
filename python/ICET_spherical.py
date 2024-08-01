import numpy as np
from vedo import *
from ipyvtklink.viewer import ViewInteractiveWidget
import time
import tensorflow as tf

# # Set CPU as available physical device (for hardware debug) ~~~~~~~~~~~~~~~~~~~~~~~~~
# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
# # tf.debugging.set_log_device_placement(True)
# tf.config.set_visible_devices([], 'GPU')
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from utils import R2Euler, Ell, jacobian_tf, R_tf, get_cluster, get_cluster_fast


class ICET():

	def __init__(self, cloud1, cloud2, fid = 30, niter = 5, draw = True, 
		x0 = tf.constant([0.0, 0.0, 0., 0., 0., 0.]), group = 2, RM = True,
		DNN_filter = False):

		self.st = time.time() #start time (for debug)

		self.min_cell_distance = 2 #begin closest spherical voxel here
		#ignore "occupied" cells with fewer than this number of pts
		self.min_num_pts = 100 #was 50 for KITTI and Ford, need to lower to 25 for CODD 
		self.fid = fid # dimension of 3D grid: [fid, fid, fid]
		self.draw = draw
		self.niter = niter
		self.alpha = 0.3 #controls alpha values when displaying ellipses
		self.DNN_filter = DNN_filter
		self.start_filter_iter = 7 #10 #iteration to start DNN rejection filter
		self.start_RM_iter = 4 #10 #iteration to start removing moving objects (set low to generate training data)
		self.DNN_thresh = 0.05 #0.03
		self.RM_thresh = 0.3 #0.125

		before = time.time()

		#load DNN
		if self.DNN_filter:
			self.model = tf.keras.models.load_model("trained_networks/KITTINet100.kmod")

		#convert cloud1 to tesnsor
		self.cloud1_tensor = tf.random.shuffle(tf.cast(tf.convert_to_tensor(cloud1), tf.float32))
		self.cloud2_tensor = tf.random.shuffle(tf.cast(tf.convert_to_tensor(cloud2), tf.float32))

		if self.draw == True:
			self.plt = Plotter(N = 1, axes = 4, bg = (1, 1, 1), interactive = True) #axis = 1
			self.disp = []
			#set vedo camera
			self.plt.camera.SetPosition( [-36.28, 13.51, 20.48] )
			self.plt.camera.SetFocalPoint( [11.18, -2.13, 0.8092] )
			self.plt.camera.SetViewUp( [0.3443, -0.124, 0.9305] )
			self.plt.camera.SetDistance( 53.70 )
			self.plt.camera.SetClippingRange( [0.18, 181.9] )
			
		#convert cloud to spherical coordinates
		self.cloud1_tensor_spherical = tf.cast(self.c2s(self.cloud1_tensor), tf.float32)
		self.cloud2_tensor_spherical = tf.cast(self.c2s(self.cloud2_tensor), tf.float32)

		#remove  points closer than minimum radial distance
		not_too_close1 = tf.where(self.cloud1_tensor_spherical[:,0] > self.min_cell_distance)[:,0]
		self.cloud1_tensor_spherical = tf.gather(self.cloud1_tensor_spherical, not_too_close1)
		self.cloud1_tensor = tf.gather(self.cloud1_tensor, not_too_close1)
		not_too_close2 = tf.where(self.cloud2_tensor_spherical[:,0] > self.min_cell_distance)[:,0]
		self.cloud2_tensor_spherical = tf.gather(self.cloud2_tensor_spherical, not_too_close2)
		self.cloud2_tensor_OG = tf.gather(self.cloud2_tensor, not_too_close2) #better to remove too close points from OG
		self.cloud2_tensor = tf.gather(self.cloud2_tensor, not_too_close2)

		self.grid_spherical( draw = False )

		self.cloud1_static = None #placeholder for returning inlier points after moving point exclusion routine

		if group == 1:
			#perform algorith using naive radial voxels
			self.main_1(niter = self.niter, x0 = x0)

		if group == 2:
			#perform algorithm with adaptive radial clustering
			self.main_2(niter = self.niter, x0 = x0, remove_moving = RM)

		if self.draw == True:
			# self.disp.append(addons.LegendBox(self.disp))
			self.plt.show(self.disp, "Spherical ICET", resetcam = False)

	def main_2(self, niter, x0, remove_moving = True):
		""" Main loop using radial clustering strategy """

		before = time.time()

		self.X = x0
		self.before_correction = x0

		# get boundaries containing useful clusters of points from first scan
		#bin points by spike
		thetamin = -np.pi
		thetamax = np.pi
		phimin =  3*np.pi/8
		phimax = 7*np.pi/8 

		edges_phi = tf.linspace(phimin, phimax, self.fid_phi)
		edges_theta = tf.linspace(thetamin, thetamax, self.fid_theta + 1)

		cloud = self.cloud1_tensor_spherical
		
		bins_theta = tfp.stats.find_bins(cloud[:,1], edges_theta)
		bins_phi = tfp.stats.find_bins(cloud[:,2], edges_phi)

		#combine bins_theta and bins_phi to get spike bins
		bins_spike = tf.cast(bins_theta*(self.fid_phi-1) + bins_phi, tf.int32)

		#save which spike each point is in to ICET object for further analysis
		self.bins_spike = bins_spike

		#find min point in each occupied spike
		occupied_spikes, idxs = tf.unique(bins_spike)
		temp =  tf.where(bins_spike == occupied_spikes[:,None]) #TODO- there has to be a better way to do this... 
		rag = tf.RaggedTensor.from_value_rowids(temp[:,1], temp[:,0])
		idx_by_rag = tf.gather(cloud[:,0], rag)

		rads = tf.transpose(idx_by_rag.to_tensor()) 
		self.rads = rads #temp for debug
		bounds = get_cluster_fast(rads, mnp = self.min_num_pts)

		corn = self.get_corners_cluster(occupied_spikes, bounds)
		inside1, npts1 = self.get_points_in_cluster(self.cloud1_tensor_spherical, occupied_spikes, bounds)	

		self.inside1 = inside1
		self.npts1 = npts1
		self.bounds = bounds
		self.occupied_spikes = occupied_spikes

		#fit gaussian
		mu1, sigma1 = self.fit_gaussian(self.cloud1_tensor, inside1, tf.cast(npts1, tf.float32))

		enough1 = tf.where(npts1 > self.min_num_pts)[:,0]
		mu1_enough = tf.gather(mu1, enough1)
		sigma1_enough = tf.gather(sigma1, enough1)

		#standard U and L method does not work with new grouping strategy
		U, L = self.get_U_and_L_cluster(sigma1_enough, mu1_enough, occupied_spikes, bounds)

		if self.draw:
			# self.visualize_L(mu1_enough, U, L)
			self.draw_ell(mu1_enough, sigma1_enough, pc = 1, alpha = self.alpha)
			self.draw_cell(corn)
			# self.draw_car()

		for i in range(niter):

			#transform cartesian point cloud 2 by estimated solution vector X
			t = self.X[:3]
			rot = R_tf(-self.X[3:])
			# self.cloud2_tensor = tf.matmul((self.cloud2_tensor_OG + t), tf.transpose(rot)) #was this in 3D-ICET paper
			self.cloud2_tensor = tf.matmul((self.cloud2_tensor_OG), tf.transpose(rot)) + t   #rotate then translate

			#convert back to spherical coordinates
			self.cloud2_tensor_spherical = tf.cast(self.c2s(self.cloud2_tensor), tf.float32)

			#find points from scan 2 that fall inside clusters
			inside2, npts2 = self.get_points_in_cluster(self.cloud2_tensor_spherical, occupied_spikes, bounds)
			#fit gaussians distributions to each of these groups of points 		
			mu2, sigma2 = self.fit_gaussian(self.cloud2_tensor, inside2, tf.cast(npts2, tf.float32))

			enough2 = tf.where(npts2 > self.min_num_pts)[:,0]
			mu2_enough = tf.gather(mu2, enough2)
			#get correspondences
			corr = tf.sets.intersection(enough1[None,:], enough2[None,:]).values
			corr_full = tf.sets.intersection(enough1[None,:], enough2[None,:]).values

			#----------------------------------------------
			if remove_moving:  
				if i >= self.start_RM_iter: #TODO: tune this to optimal value
					print("\n ---checking for moving objects---")
					
					#test - re-calculting this here
					y0_i_full = tf.gather(mu1, corr_full)
					y_i_full = tf.gather(mu2, corr_full)

					self.residuals_full = y_i_full - y0_i_full
				
					# #------------------------------------------------------------------------------------------------
					# #Using binned mode oulier exclusion (get rid of everything outside of some range close to 0)
					# nbins = 30
					# edges = tf.linspace(-0.75, 0.75, nbins)
					# bins_soln = tfp.stats.find_bins(self.residuals_full[:,0], edges)
					# bad_idx = tf.where(bins_soln != (nbins//2 - 1))[:,0][None, :]

					# bins_soln2 = tfp.stats.find_bins(self.residuals_full[:,1], edges)
					# bad_idx2 = tf.where(bins_soln2 != (nbins//2 - 1))[:,0][None, :]
					# bad_idx = tf.sets.union(bad_idx, bad_idx2).values
					# #------------------------------------------------------------------------------------------------

					# #------------------------------------------------------------------------------------------------
					# #Using Gaussian n-sigma outlier exclusion on translation

					# metric1 = self.residuals_full[:,0]
					# metric2 = self.residuals_full[:,1]
					# mu_x = tf.math.reduce_mean(metric1)
					# sigma_x = tf.math.reduce_std(metric1)
					
					# # #just x------------
					# # bad_idx = tf.where( tf.math.abs(metric1) > mu_x + 2*sigma_x )[:, 0]
					# # #------------------

					# #x and y---------
					# bad_idx = tf.where( tf.math.abs(metric1) > mu_x + 2.0*sigma_x )[:,0][None, :]
					# # print(" \n bad_idx1", bad_idx)

					# mu_y = tf.math.reduce_mean(metric2)
					# sigma_y = tf.math.reduce_std(metric2)
					# bad_idx2 = tf.where( tf.math.abs(metric2) > mu_y + 	2.0*sigma_y )[:,0][None, :]
					# # print("\n bad_idx2", bad_idx2)
					# bad_idx = tf.sets.union(bad_idx, bad_idx2).values

					# #if using rotation too
					# # self.bad_idx = bad_idx
					# # self.bad_idx_rot = bad_idx_rot
					# # bad_idx = tf.sets.union(bad_idx[None, :], bad_idx_rot[None, :]).values 
					# #-----------------

					# # print("corr \n", corr)
					# print("bad idx", bad_idx)
					# # print(tf.gather(it.dx_i[:,0], bad_idx))
					# # print(tf.gather(occupied_spikes, corr))
					# #------------------------------------------------------------------------------------------------

					#hard cutoff for outlier rejection
					#NEW (5/7)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
					both = tf.sets.intersection(enough1[None,:], corr_full[None,:]).values
					#get indices of mu1 that correspond to mu2 that also have sufficient number of points
					ans = tf.where(enough1[:,None] == both)[:,0]
					
					#test moving these here
					U_i = tf.gather(U, ans)
					U_iT = tf.transpose(U_i, [0,2,1])
					L_i = tf.gather(L, ans)
					# residuals_compact = L_i @ U_i @ tf.gather(self.residuals_full[:,:,None], corr_full) #was this (incorrect)
					# residuals_compact = L_i @ U_iT @ tf.gather(self.residuals_full[:,:,None], ans) #(5/19) -> debug: should this be U_i or U_iT?
					residuals_compact = L_i @ U_iT @ self.residuals_full[:,:,None] #correct (5/20)

					# self.RM_thresh = 0.03 #0.1 #0.05
					# bidx = tf.where(residuals_compact > thresh )[:,0] #TODO: consider absolute value!
					bidx = tf.where(tf.math.abs(residuals_compact) > self.RM_thresh )[:,0]
					# print(residuals_compact)
					bad_idx = bidx
					# print("bad_idx", bidx)
					#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

					# #------------------------------------------------------------------------------------------------
					# Compare rotation about the vertical axis between each distribution correspondance
					s1 = tf.transpose(tf.gather(sigma1, corr), [1, 2, 0])
					s2 = tf.transpose(tf.gather(sigma2, corr), [1, 2, 0])

					self.angs1 = R2Euler(s1)[2,:]
					self.angs2 = R2Euler(s2)[2,:]

					self.res = self.angs1 - self.angs2

					mean = np.mean(self.res)
					std = np.std(self.res)
					# bad_idx_rot = tf.where(np.abs(self.res) > mean + 1*std )[:, 0]

					cutoff = 0.1 #0.1
					bad_idx_rot = tf.where(np.abs(self.res) > cutoff)[:, 0]

					# print("bad_idx_rot", bad_idx_rot)

					bad_idx = tf.sets.union(bad_idx[None, :], bad_idx_rot[None, :]).values
					# # #------------------------------------------------------------------------------------------------


					bounds_bad = tf.gather(bounds, tf.gather(corr, bad_idx))
					bad_idx_corn_moving = self.get_corners_cluster(tf.gather(occupied_spikes, tf.gather(corr, bad_idx)), bounds_bad)

					ignore_these = tf.gather(corr, bad_idx)
					corr = tf.sets.difference(corr[None, :], ignore_these[None, :]).values

					#temp
					self.U_i = U_i
					self.L_i = L_i

			#----------------------------------------------

			#----------------------------------------------
			#Use DNN to remove cells affected by persepective shift
			if self.DNN_filter and i >= self.start_filter_iter:
				#DEBUG (5/2)- replaced all references of <corr> to <corr_full>

				nSamplePts = 100 #50

				print("\n ---checking for perspective shift---")
				#get indices of rag with >= 25 elements
				ncells = tf.shape(corr_full)[0].numpy() #num of voxels with sufficent number of points
				#Get ragged tensor containing all points from each scan inside each sufficient voxel
				en1 = tf.gather(inside1, corr_full)
				en2 = tf.gather(inside2, corr_full)

				#init array to store indices
				idx1 = np.zeros([ncells ,nSamplePts])
				idx2 = np.zeros([ncells ,nSamplePts])

				#loop through each element of ragged tensor
				for i in range(ncells):
				    idx1[i,:] = tf.random.shuffle(en1[i])[:nSamplePts].numpy() #shuffle order and take first 25 elements
				    idx2[i,:] = tf.random.shuffle(en2[i])[:nSamplePts].numpy() #shuffle order and take first 25 elements

				idx1 = tf.cast(tf.convert_to_tensor(idx1), tf.int32)
				idx2 = tf.cast(tf.convert_to_tensor(idx2), tf.int32)

				from1 = tf.gather(self.cloud1_tensor, idx1)
				from2 = tf.gather(self.cloud2_tensor, idx2)

				x_test = tf.concat((from1, from2), axis = 1)

				# #One shot~~~~~~~~~~~
				# dnnsoln = self.model.predict(x_test)
				# dnnsoln = tf.convert_to_tensor(dnnsoln)
				# #~~~~~~~~~~~~~~~~~~~

				#Iterative~~~~~~~~~~~~
				correction = 0
				niter = 2
				inputs = x_test
				for _ in range(niter):
					correction += self.model.predict(inputs) #was this for KITTI/ Ford trained model
					# correction += 0.1*self.model.predict(inputs) #need to scale if model trained on MatLab data
					from1 = np.array([from1[:,:,0] + correction[:,0][:,None], from1[:,:,1] + correction[:,1][:,None], from1[:,:,2] + correction[:,2][:,None]])
					from1 = np.transpose(from1, (1,2,0))
					inputs = np.append(from1, from2, axis = 1)

				dnnsoln = tf.convert_to_tensor(correction)
				#~~~~~~~~~~~~~~~~~~~~

				#was this - DEBUG: think it may be creating a chicken and egg problem
				# icetsoln = tf.gather(self.residuals, corr)
				icetsoln = tf.math.reduce_mean(tf.gather(self.cloud1_tensor,en1), axis = 1) - tf.math.reduce_mean(tf.gather(self.cloud2_tensor,en2), axis = 1)

				#Signs need to be flipped
				icetsoln = -icetsoln
				dnnsoln = -dnnsoln

				# print(enough1)
				# print(self.corr)

				both = tf.sets.intersection(enough1[None,:], corr_full[None,:]).values
				ans = tf.where(enough1[:,None] == both)[:,0]
				
				U_i = tf.gather(U, ans)
				L_i = tf.gather(L, ans)
				U_i_dnn = U_i
				LUT = tf.matmul(L_i, tf.transpose(U_i, [0,2,1]))
				dz_new = tf.matmul(LUT, dnnsoln[:,:,None])
				it_compact = tf.matmul(LUT, icetsoln[:,:,None])
				it_compact_xyz = tf.matmul(U_i, it_compact)
				dnn_compact = tf.matmul(LUT, dnnsoln[:,:,None])
				dnn_compact_xyz = tf.matmul(U_i, dnn_compact)

				# TEST 11/9/22 - remove ambiguous axis suppression of DNN solution on non-oblate clouds------
				# dnn_compact_xyz = dnnsoln[:,:,None]
				#--------------------------------------------------------------------------------------------

				#find where the largest difference in residuals are
				bad_idx = tf.where(tf.math.abs(it_compact_xyz - dnn_compact_xyz) > self.DNN_thresh)[:,0]
				bad_idx = tf.unique(bad_idx)[0] #get rid of repeated indices
				# print("bad_idx", bad_idx)

				good_idx = tf.where(tf.math.abs(it_compact_xyz - dnn_compact_xyz) < self.DNN_thresh)[:,0]
				good_idx = tf.unique(good_idx)[0] #get rid of repeated indices
				# print("good_idx", good_idx)
				# print(tf.math.abs(it_compact_xyz - dnn_compact_xyz))

				#draw bad cells
				bounds_bad = tf.gather(self.bounds, tf.gather(corr_full, bad_idx))
				bad_idx_corn_DNN = self.get_corners_cluster(tf.gather(self.occupied_spikes, tf.gather(corr_full, bad_idx)), bounds_bad)

				#remove perspective shifts from corr
				ignore_these_dnn = tf.gather(corr_full, bad_idx)
				#TODO- there may be a bug here
				corr = tf.sets.difference(corr[None,:], ignore_these_dnn[None,:]).values #was this
			
				idx_to_draw_dnn_soln = tf.gather(mu1_enough, ans)
				# self.draw_DNN_soln(dnn_compact_xyz[:,:,0], it_compact_xyz[:,:,0], tf.gather(mu1_enough, ans))

				# print("\n ~~~~~~~~~~~~~~ \n DNN Filter", time.time() - before, "\n total: ",  time.time() - self.st, "\n ~~~~~~~~~~~~~~")
				before = time.time()
			#----------------------------------------------
			
			#for MC simulations
			if i < self.start_filter_iter:
				self.before_correction = self.X

			y0_i_full = tf.gather(mu1, corr_full)
			y_i_full = tf.gather(mu2, corr_full)

			y0_i = tf.gather(mu1, corr)
			sigma0_i = tf.gather(sigma1, corr)
			npts0_i = tf.gather(npts1, corr)

			y_i = tf.gather(mu2, corr)
			sigma_i = tf.gather(sigma2, corr)
			npts_i = tf.gather(npts2, corr)
			
			#need special indexing for U_i and L_i since they are derived from <mu1_enough>
			# rather than the full mu1 tensor:
			#  1) get IDX of elements that are in both enough1 and corr
			#  2) use this to index U and L to get U_i and L_i
			both = tf.sets.intersection(enough1[None,:], corr[None,:]).values
			ans = tf.where(enough1[:,None] == both)[:,0]			
			U_i = tf.gather(U, ans)
			L_i = tf.gather(L, ans)

			#hold on to these values inside ICET object so we can use these to compare results with DNN
			self.corr = corr
			self.residuals = y_i - y0_i
			self.U = U_i
			self.L = L_i
			self.corn = tf.gather(corn, corr)

			#get matrix containing partial derivatives for each voxel mean
			H = jacobian_tf(tf.transpose(y_i), self.X[3:]) # shape = [num of corr * 3, 6]
			H = tf.reshape(H, (tf.shape(H)[0]//3,3,6)) # -> need shape [#corr//3, 3, 6]

			U_iT = tf.transpose(U_i, [0,2,1])
			L_iT = tf.transpose(L_i, [0,2,1])

			#construct sensor noise covariance matrix
			R_noise = (tf.transpose(tf.transpose(sigma0_i, [1,2,0]) / tf.cast(npts0_i - 1, tf.float32)) + 
						tf.transpose(tf.transpose(sigma_i, [1,2,0]) / tf.cast(npts_i - 1, tf.float32)) )
			#use projection matrix to remove extended directions
			R_noise = L_i @ U_iT @ R_noise @ U_i @ L_iT #correct(?)

			#take inverse of R_noise to get our weighting matrix
			W = tf.linalg.pinv(R_noise)

			# use LUT to remove rows of H corresponding to overly extended directions
			LUT = L_i @ U_iT
			H_z = LUT @ H

			HTWH = tf.math.reduce_sum(tf.matmul(tf.matmul(tf.transpose(H_z, [0,2,1]), W), H_z), axis = 0) #was this (which works)
			HTW = tf.matmul(tf.transpose(H_z, [0,2,1]), W)

			L2, lam, U2 = self.check_condition(HTWH)
			self.lam = lam # temp -- hold on to lam for later analyis in signage dataset

			# create alternate corrdinate system to align with axis of scan 1 distributions
			z = tf.squeeze(tf.matmul(LUT, y_i[:,:,None]))
			z0 = tf.squeeze(tf.matmul(LUT, y0_i[:,:,None]))	
			dz = z - z0
			dz = dz[:,:,None] #need to add an extra dimension to dz to get the math to work out

			dx = tf.squeeze(tf.matmul( tf.matmul(tf.linalg.pinv(L2 @ lam @ tf.transpose(U2)) @ L2 @ tf.transpose(U2) , HTW ), dz))	
			dx = tf.math.reduce_sum(dx, axis = 0)
			self.X += dx

			print("\n estimated solution vector X: \n", self.X)

			#get output covariance matrix
			self.Q = tf.linalg.pinv(HTWH)
			self.pred_stds = tf.linalg.tensor_diag_part(tf.math.sqrt(tf.abs(self.Q)))

		print("pred_stds: \n", self.pred_stds)

		#draw PC2
		if self.draw == True:
			if self.DNN_filter:
				self.draw_cell(bad_idx_corn_DNN, bad = 2)
				# self.draw_DNN_soln(dnn_compact_xyz[:,:,0], it_compact_xyz[:,:,0], idx_to_draw_dnn_soln) #just in compact directions
				self.draw_DNN_soln(dnnsoln, icetsoln, idx_to_draw_dnn_soln) #raw solutions
			if remove_moving:
				self.draw_cell(bad_idx_corn_moving, bad = True) #for debug

			self.draw_ell(y_i, sigma_i, pc = 2, alpha = self.alpha)
			self.draw_cloud(self.cloud1_tensor.numpy(), pc = 1)
			self.draw_cloud(self.cloud2_tensor.numpy(), pc = 2)
			# self.draw_correspondences(mu1, mu2, corr_full) #displays used correspondences
			# self.visualize_L(tf.gather(mu1_enough, ans), U_i, L_i)

		if remove_moving == True:
			#hold on to points from scan2 that do not contain moving objects
			idx_of_points_inside_good_clusters = tf.gather(inside2, corr)
			self.cloud2_static = tf.gather(self.cloud2_tensor_OG, idx_of_points_inside_good_clusters).values.numpy()
		else:
			self.cloud2_static = self.cloud2_tensor_OG.numpy()


	def get_points_in_cluster(self, cloud, occupied_spikes, bounds):
		""" returns ragged tensor containing the indices of points in <cloud> in each cluster 
		
			cloud = point cloud tensor
			occupied_spikes = tensor containing idx of spikes corresponding to bounds
			bounds = tensor containing min and max radius for each occupied spike

		"""
		thetamin = -np.pi
		thetamax = np.pi
		phimin =  3*np.pi/8
		phimax = 7*np.pi/8

		edges_phi = tf.linspace(phimin, phimax, self.fid_phi) #was this for regular cells
		bins_phi = tfp.stats.find_bins(cloud[:,2], edges_phi)

		edges_theta = tf.linspace(thetamin, thetamax, self.fid_theta + 1)
		bins_theta = tfp.stats.find_bins(cloud[:,1], edges_theta)

		spike_idx = tf.cast(bins_theta*(self.fid_phi-1) + bins_phi, tf.int32)

		#get idx of spike for each applicable point
		cond1 = spike_idx == occupied_spikes[:,None] #match spike IDs
		cond2 = cloud[:,0] < tf.cast(bounds[:,1][:,None], tf.float32) #closer than max bound
		cond3 = cloud[:,0] > tf.cast(bounds[:,0][:,None], tf.float32) #further than min bound

		inside1 = tf.where(tf.math.reduce_all(tf.Variable([cond1, cond2, cond3]), axis = 0))
		numPtsPerCluster = tf.math.bincount(tf.cast(inside1[:,0], tf.int32))
		inside1 = tf.RaggedTensor.from_value_rowids(inside1[:,1], inside1[:,0])

		return(inside1, numPtsPerCluster)

	def get_U_and_L_cluster(self, sigma1, mu1, occupied_spikes, bounds):
		""" get U and L when using cluster point grouping """ 


		eigenval, eigenvec = tf.linalg.eig(sigma1)
		U = tf.math.real(eigenvec) #was this

		#need to create [N,3,3] diagonal matrices for axislens
		zeros = tf.zeros([tf.shape(tf.math.real(eigenval))[0]])
		axislen = tf.Variable([tf.math.real(eigenval)[:,0], zeros, zeros,
							   zeros, tf.math.real(eigenval)[:,1], zeros,
							   zeros, zeros, tf.math.real(eigenval)[:,2]])

		axislen = tf.reshape(tf.transpose(axislen), (tf.shape(axislen)[1], 3, 3)) #variance not std...?

		# get projections of axis length in each direction
		rotated = tf.matmul(axislen, tf.transpose(U, [0, 2, 1])) #new

		# axislen_actual = 2*tf.math.sqrt(axislen) #theoretically correct
		axislen_actual = 3*tf.math.sqrt(axislen) #was this (works best with one edge extended detection criteria)
		# axislen_actual = 0.1*tf.math.sqrt(axislen) #turns off extended axis pruning

		rotated_actual = tf.matmul(axislen_actual, tf.transpose(U, [0, 2, 1]))
	
		#get points at the ends of each distribution ellipse
		mu_repeated = tf.tile(mu1, [3,1])
		mu_repeated = tf.reshape(tf.transpose(mu_repeated), [3,3,-1])
		mu_repeated = tf.transpose(mu_repeated, [2,1,0])
		mu_repeated = tf.reshape(mu_repeated, [-1,3, 3])

		P1 = mu_repeated + rotated_actual
		P1 = tf.reshape(P1, [-1, 3])
		P2 = mu_repeated - rotated_actual
		P2 = tf.reshape(P2, [-1, 3])

		#Assumes mu is always going to be inside the corresponding cell (should almost always be the case, if not, its going to fail anyways)
		insideP_ideal, nptsP_ideal = self.get_points_in_cluster(self.c2s(tf.reshape(mu_repeated, [-1,3])), occupied_spikes, bounds)
		insideP_ideal = insideP_ideal.to_tensor(default_value = -1)

		#find which points in P are actually inside which cell in <cells>
		insideP1_actual, nptsP1_actual = self.get_points_in_cluster(self.c2s(P1), occupied_spikes, bounds)
		insideP1_actual = insideP1_actual.to_tensor(shape = tf.shape(insideP_ideal), default_value = -1) #force to be same size as insideP_ideal
		insideP2_actual, nptsP2_actual = self.get_points_in_cluster(self.c2s(P2), occupied_spikes, bounds)
		insideP2_actual = insideP2_actual.to_tensor(shape = tf.shape(insideP_ideal), default_value = -1)

		#compare the points inside each cell to how many there are supposed to be
		#	(any mismatch signifies an overly extended direction)
		outside1 = tf.sets.intersection(insideP_ideal, insideP1_actual).values
		outside2 = tf.sets.intersection(insideP_ideal, insideP2_actual).values

		#combine both the positive and negative axis directions
		# outside_both = tf.cast(tf.sets.intersection(outside1[None, :], outside2[None, :]).values[:,None], tf.int32) # only need one edge outside cell (was this)
		outside_both = tf.cast(tf.sets.union(outside1[None, :], outside2[None, :]).values[:,None], tf.int32) #both edeges need to be outside cell to be ambigous
		# print("unambiguous indices", outside_both)

		data = tf.ones((tf.shape(outside_both)[0],3))
		I = tf.tile(tf.eye(3), (tf.shape(U)[0], 1))

		mask = tf.scatter_nd(indices = outside_both, updates = data, shape = tf.shape(I))

		L = mask * I
		L = tf.reshape(L, (tf.shape(L)[0]//3,3,3))

		return(U, L)

			
	def get_U_and_L(self, sigma1, mu1, cells, method = 0):
		""" 	sigma1 = sigmas from the first scan
				cells = tensor containing the indices of each scan
				
				method == 0: old method simiar to 3D-ICET
				method == 1: New "unsceneted KF" strategy

				U = rotation matrix for each voxel to transform scan 2 distribution
				 into frame corresponding to ellipsoid axis in keyframe
			    L = matrix to prune extended directions in each voxel (from keyframe)
			    """

		eigenval, eigenvec = tf.linalg.eig(sigma1)
		U = tf.math.real(eigenvec)

		#need to create [N,3,3] diagonal matrices for axislens
		zeros = tf.zeros([tf.shape(tf.math.real(eigenval))[0]])
		axislen = tf.Variable([tf.math.real(eigenval)[:,0], zeros, zeros,
							   zeros, tf.math.real(eigenval)[:,1], zeros,
							   zeros, zeros, tf.math.real(eigenval)[:,2]])

		axislen = tf.reshape(tf.transpose(axislen), (tf.shape(axislen)[1], 3, 3)) #variance not std...?
		# print("\n axislen \n", axislen)

		#new method (UKF-type strategy)
		#_______________________________________________________________________________
		if method == 1:
			# get projections of axis length in each direction
			rotated = tf.matmul(axislen, tf.transpose(U, [0, 2, 1])) #new

			#QUESTION: should I scale this up if we use stretched voxels?
			axislen_actual = 2*tf.math.sqrt(axislen)
			# print(axislen_actual)
			rotated_actual = tf.matmul(axislen_actual, tf.transpose(U, [0, 2, 1]))
			# print("rotated_actual", rotated_actual)
		
			#get points at the ends of each distribution ellipse
			# print("mu1", mu1)
			mu_repeated = tf.tile(mu1, [3,1])
			mu_repeated = tf.reshape(tf.transpose(mu_repeated), [3,3,-1])
			mu_repeated = tf.transpose(mu_repeated, [2,1,0])
			mu_repeated = tf.reshape(mu_repeated, [-1,3, 3])
			# print("mu_repeated", mu_repeated)

			P1 = mu_repeated + rotated_actual
			P1 = tf.reshape(P1, [-1, 3])
			P2 = mu_repeated - rotated_actual
			P2 = tf.reshape(P2, [-1, 3])

			#draw tempoary marking at boundaries of each distribution ellipse
			# self.disp.append(Points(P1.numpy(), 'g', r = 10))
			# self.disp.append(Points(P2.numpy(), 'g', r = 10))

			#find out which points in P SHOULD be inside each cell
			insideP_ideal, nptsP_ideal = self.get_points_inside(self.c2s(tf.reshape(mu_repeated, [-1,3])), cells[:,None])
			insideP_ideal = insideP_ideal.to_tensor()
			# print("insideP_ideal", insideP_ideal)

			#find which points in P are actually inside which cell in <cells>
			insideP1_actual, nptsP1_actual = self.get_points_inside(self.c2s(P1), cells[:, None])
			# print(insideP1_actual)
			insideP1_actual = insideP1_actual.to_tensor(shape = tf.shape(insideP_ideal)) #force to be same size as insideP_ideal
			# print("insideP1_actual", insideP1_actual)
			insideP2_actual, nptsP2_actual = self.get_points_inside(self.c2s(P2), cells[:, None])
			insideP2_actual = insideP2_actual.to_tensor(shape = tf.shape(insideP_ideal))
			# print("insideP2_actual", insideP2_actual)

			#compare the points inside each cell to how many there are supposed to be
			#	(any mismatch signifies an overly extended direction)
			outside1 = tf.sets.intersection(insideP_ideal, insideP1_actual).values
			# print("\n outside1", outside1)
			outside2 = tf.sets.intersection(insideP_ideal, insideP2_actual).values
			# print("\n outside2", outside2)

			#combine both the positive and negative axis directions
			outside_both = tf.cast(tf.sets.intersection(outside1[None, :], outside2[None, :]).values[:,None], tf.int32)
			# print("unambiguous indices", outside_both)

			data = tf.ones((tf.shape(outside_both)[0],3))
			I = tf.tile(tf.eye(3), (tf.shape(U)[0], 1))

			mask = tf.scatter_nd(indices = outside_both, updates = data, shape = tf.shape(I))

			L = mask * I
			L = tf.reshape(L, (tf.shape(L)[0]//3,3,3))

		#_______________________________________________________________________________


		#old method - just consider principal axis length
		if method == 0:

			# get projections of axis length in each direction
			rotated = tf.abs(tf.matmul(U,axislen)) #was this pre 3/10
			# print("rotated", rotated)

			#need information on the cell index to be able to perform truncation 
			#	-> (cells further from vehicle will require larger distribution length thresholds)
			shell = cells//(self.fid_theta*(self.fid_phi - 1))
			# print("shell", shell)
			r_grid, _ = tf.unique(self.grid[:,0])
			# print("r_grid", r_grid)
			cell_width = tf.experimental.numpy.diff(r_grid)
			# print("cell_width", cell_width)
			# thresholds = (tf.gather(cell_width, shell)**2)/32
			# thresholds = (tf.gather(cell_width, shell)**2)/64 #was this
			thresholds = (tf.gather(cell_width, shell)**2) #NDT override


			#tile to so that each threshold is repeated 3 times (for each axis)
			thresholds = tf.reshape(tf.transpose(tf.reshape(tf.tile(thresholds[:,None], [3,1]), [3,-1])), [-1,3])[:,None]
			# print("thresholds", thresholds)

			greater_than_thresh = tf.math.greater(rotated, thresholds)
			# print(greater_than_thresh)
			ext_idx = tf.math.reduce_any(greater_than_thresh, axis = 1) #was this
			# ext_idx = tf.math.reduce_any(greater_than_thresh, axis = 2) #test
			compact = tf.where(tf.math.reduce_any(tf.reshape(ext_idx, (-1,1)), axis = 1) == False)
			compact =  tf.cast(compact, tf.int32)
			# print("compact", compact)
			data = tf.ones((tf.shape(compact)[0],3))
			I = tf.tile(tf.eye(3), (tf.shape(U)[0], 1))

			mask = tf.scatter_nd(indices = compact, updates = data, shape = tf.shape(I))

			L = mask * I
			L = tf.reshape(L, (tf.shape(L)[0]//3,3,3))

		return(U,L)

	def visualize_L(self, y0, U, L):
		""" for each voxel center, mu, this func draws untruncated axis via L 
			transformed into the frame of the distribution ellipsoids via U  """

		for i in range(tf.shape(y0)[0]):

			ends =  L[i] @ tf.transpose(U[i])
			# ends =  L[i] @ U[i] #WRONG!!


			arrow_len = 0.5
			arr1 = shapes.Arrow(y0[i].numpy(), (y0[i] + arrow_len * ends[0,:]).numpy(), c = 'red')
			self.disp.append(arr1)
			arr1 = shapes.Arrow(y0[i].numpy(), (y0[i] - arrow_len * ends[0,:]).numpy(), c = 'red')
			self.disp.append(arr1)

			arr2 = shapes.Arrow(y0[i].numpy(), (y0[i] + arrow_len * ends[1,:]).numpy(), c = 'green')
			self.disp.append(arr2)
			arr2 = shapes.Arrow(y0[i].numpy(), (y0[i] - arrow_len * ends[1,:]).numpy(), c = 'green')
			self.disp.append(arr2)
			
			arr3 = shapes.Arrow(y0[i].numpy(), (y0[i] + arrow_len * ends[2,:]).numpy(), c = 'blue')
			self.disp.append(arr3)
			arr3 = shapes.Arrow(y0[i].numpy(), (y0[i] - arrow_len * ends[2,:]).numpy(), c = 'blue')
			self.disp.append(arr3)

	def draw_DNN_soln(self, dnnsoln, itsoln, mu1):
		""" For each qualifying voxel, draw the solution vector estimated by the scan registation DNN 

			#dnnsoln = [n, 3] tensor with x, y, z translation estimates for each voxel
			itsoln = [n, 3] tensor, used to debug places where ICET and DNN solns differ greatly, want
							to make sure this works the same as our other perspective shift id technique
			mu1 = distribution centers from scan1 (only where sufficient correspondences occur)
			"""

		for i in range(tf.shape(dnnsoln)[0].numpy()):
			#normalize len of each arrow
			# arrowlen = 1/(np.sqrt(dnnsoln[i,0].numpy()**2 + dnnsoln[i,1].numpy()**2 + dnnsoln[i,2].numpy()**2))
			arrowlen = 1 #leave arrows proportional to residual distance
			# A = Arrow2D(startPoint = mu1[i].numpy(), endPoint = mu1[i].numpy() + arrowlen*dnnsoln[i,:].numpy(), c = 'purple')
			A = shapes.Arrow(mu1[i].numpy(), mu1[i].numpy() + arrowlen*dnnsoln[i,:].numpy(), c = 'purple')
			self.disp.append(A)

			#Draw ICET solns as well (for debug)
			# arrowlen = 1/(np.sqrt(itsoln[i,0].numpy()**2 + itsoln[i,1].numpy()**2 + itsoln[i,2].numpy()**2))
			B = shapes.Arrow(mu1[i].numpy(), mu1[i].numpy() + arrowlen*itsoln[i,:].numpy(), c = 'yellow')
			self.disp.append(B)


			# #draw big dot if dnnsoln and itsoln disagree
			# if (abs((dnnsoln[i] - itsoln[i]).numpy()) > 0.1).any():
			# 	# print(i)
			# 	dot = Points(np.array([[mu1[i,0].numpy(), mu1[i,1].numpy(), mu1[i,2].numpy()]]), c = "purple", r = 20)
			# 	self.disp.append(dot)


	def check_condition(self, HTWH):
		"""verifies that HTWH is invertable and if not, 
			reduces dimensions to make inversion possible

			L2 = identity matrix which keeps non-extended axis of solution
			lam = diagonal eigenvalue matrix
			U2 = rotation matrix to transform for L2 pruning 
			"""

		cutoff = 1e7 # best value for KITTI dataset

		#do eigendecomposition
		eigenval, eigenvec = tf.linalg.eig(HTWH)
		eigenval = tf.math.real(eigenval)
		eigenvec = tf.math.real(eigenvec)

		#sort eigenvals by size -default sorts small to big
		# small2big = tf.sort(eigenval)

		#test if condition number is bigger than cutoff
		condition = eigenval[-1] / eigenval[0]

		everyaxis = tf.cast(tf.linspace(0,5,6), dtype=tf.int32)
		remainingaxis = everyaxis
		i = tf.Variable([0],dtype = tf.int32) #count var
		#loop until condition number is small enough to make matrix invertable
		while abs(condition) > cutoff:

			condition = eigenval[-1] / tf.gather(eigenval, i)

			if abs(condition) > cutoff:
				i.assign_add(tf.Variable([1],dtype = tf.int32))
				remainingaxis = everyaxis[i.numpy()[0]:]

		#create identity matrix truncated to only have the remaining axis
		L2 = tf.gather(tf.eye(6), remainingaxis)

		# #alternate strategy- zero out instead of keeping axis truncated
		# while tf.shape(L2)[0] < 6:
		# 	L2 = tf.concat((tf.zeros([1,6]), L2), axis = 0)

		# print("\n L2 \n", L2)

		U2 = eigenvec
		# print("\n U2^T \n", tf.transpose(U2))

		#TODO: scale eigenvectors associated with rotational components of solution

		lam = tf.eye(6)*eigenval

		return(L2, lam, U2)


	def draw_ell(self, mu, sigma, pc = 1, alpha = 1):
		"""draw distribution ellipses given mu and sigma tensors"""

		if pc == 1:
			color = [0.8, 0.3, 0.3]
		if pc ==2:
			color = [0.3, 0.3, 0.8]

		for i in range(tf.shape(sigma)[0]):

			eig = np.linalg.eig(sigma[i,:,:].numpy())
			eigenval = eig[0] #correspond to lengths of axis
			eigenvec = eig[1]

			# assmues decreasing size
			a1 = eigenval[0]
			a2 = eigenval[1]
			a3 = eigenval[2]

			if mu[i,0] != 0 and mu[i,1] != 0:
				ell = Ell(pos=(mu[i,0], mu[i,1], mu[i,2]), axis1 = 4*np.sqrt(abs(a1)), 
					axis2 = 4*np.sqrt(abs(a2)), axis3 = 4*np.sqrt(abs(a3)), 
					angs = (np.array([-R2Euler(eigenvec)[0], -R2Euler(eigenvec)[1], -R2Euler(eigenvec)[2] ])), c=color, alpha=alpha, res=12)
				
				self.disp.append(ell)

	def draw_correspondences(self, mu1, mu2, corr):
		""" draw arrow between distributions between scans that:
			1- contain sufficient number of points 
			2- occupy the same voxel """

		# print("correspondences", corr)
		for i in corr:
			a = shapes.Arrow(mu2[i].numpy(), mu1[i].numpy(), c = "black") #s = 0.01 #for thick lines
			self.disp.append(a)


	def get_corners_cluster(self, occupied_spikes, bounds):
		""" get 8 corners of region bounded by spike IDs and radial bounds """

		#spike IDs are the same as as cell IDs for the innermost shell of self.grid
		# 		so we can use that to get the theta and phi components
		corn = self.get_corners(occupied_spikes).numpy()
		# print("corn temp\n", np.shape(corn))

		#replace rad in grid with bounds
		corn[:,0,0] = bounds[:,0].numpy()
		corn[:,1,0] = bounds[:,0].numpy()
		corn[:,2,0] = bounds[:,1].numpy()
		corn[:,3,0] = bounds[:,1].numpy()
		corn[:,4,0] = bounds[:,0].numpy()
		corn[:,5,0] = bounds[:,0].numpy()
		corn[:,6,0] = bounds[:,1].numpy()
		corn[:,7,0] = bounds[:,1].numpy()

		return(corn)


	def get_corners(self, cells, tophat = 0):
		""" returns  spherical coordinates of coners of each input cell 
			cells = tensor containing cell indices """

		#to account for wrapping around at end of each ring
		per_shell = self.fid_theta*(self.fid_phi - 1) #number of cells per radial shell
		fix =  (self.fid_phi*self.fid_theta)*((((cells)%per_shell) + (self.fid_phi-1) )//per_shell)
		n = cells + cells//(self.fid_phi - 1)

		if tophat == 1:
			g = self.grid_tophat
		else:
			g = self.grid

		p1 = tf.gather(g, n)
		p2 = tf.gather(g, n+self.fid_phi - fix)
		p3 = tf.gather(g, n + self.fid_theta*self.fid_phi)
		p4 = tf.gather(g, n + self.fid_phi + (self.fid_theta*self.fid_phi) - fix)
		p5 = tf.gather(g, n + 1)
		p6 = tf.gather(g, n+self.fid_phi +1 - fix)
		p7 = tf.gather(g, n + (self.fid_theta*self.fid_phi) + 1)
		p8 = tf.gather(g, n + self.fid_phi + (self.fid_theta*self.fid_phi) +1 - fix)

		out = tf.transpose(tf.Variable([p1, p2, p3, p4, p5, p6, p7, p8]), [1, 0, 2])

		return(out)


	def fit_gaussian(self, cloud, rag, npts):
		""" fits 3D gaussian distribution to each elelment of 
			rag, which cointains indices of points in cloud """

		coords = tf.gather(cloud, rag)

		xpos = tf.gather(cloud[:,0], rag)
		ypos = tf.gather(cloud[:,1], rag)
		zpos = tf.gather(cloud[:,2], rag)
		# print("mux",mu[:,0])
		# print("took", time.time()-st, "s to tf.gather")
		st = time.time()

		#if GPU is not available
		if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
			#old (works on CPU but SLOW)
			mu = tf.math.reduce_mean(coords, axis=1)
			# print("mu", tf.shape(mu))
			# print("mu[:,0][:,None]", tf.shape(mu[:,0][:,None]))
			# print("xpos", tf.shape(xpos))
			xx = tf.math.reduce_sum(tf.math.square(xpos - mu[:,0][:,None] ), axis = 1)/npts
			yy = tf.math.reduce_sum(tf.math.square(ypos - mu[:,1][:,None] ), axis = 1)/npts
			zz = tf.math.reduce_sum(tf.math.square(zpos - mu[:,2][:,None] ), axis = 1)/npts
			xy = tf.math.reduce_sum( (xpos - mu[:,0][:,None])*(ypos - mu[:,1][:,None]), axis = 1)/npts  #+
			xz = tf.math.reduce_sum( (xpos - mu[:,0][:,None])*(zpos - mu[:,2][:,None]), axis = 1)/npts #-
			yz = tf.math.reduce_sum( (ypos - mu[:,1][:,None])*(zpos - mu[:,2][:,None]), axis = 1)/npts #-
			sigma = tf.Variable([xx, xy, xz,
								 xy, yy, yz,
								 xz, yz, zz]) 
			sigma = tf.reshape(tf.transpose(sigma), (tf.shape(sigma)[1] ,3,3))
			return(mu, sigma)

		#if GPU is available
		else:
			#new method downsampling to first n points in each ragged tensor -- MUCH FASTER (but only works on GPU)
			mu = tf.math.reduce_mean(coords, axis = 1)[:,None]
			idx = tf.range(self.min_num_pts)
			# idx = tf.range(self.min_num_pts-1) #test
			xpos = tf.gather(xpos, idx, axis = 1)
			ypos = tf.gather(ypos, idx, axis = 1)
			zpos = tf.gather(zpos, idx, axis = 1)

			xx = tf.math.reduce_sum(tf.math.square(xpos - mu[:,:,0] ), axis = 1)/self.min_num_pts
			yy = tf.math.reduce_sum(tf.math.square(ypos - mu[:,:,1] ), axis = 1)/self.min_num_pts
			zz = tf.math.reduce_sum(tf.math.square(zpos - mu[:,:,2] ), axis = 1)/self.min_num_pts
			xy = tf.math.reduce_sum( (xpos - mu[:,:,0])*(ypos - mu[:,:,1]), axis = 1)/self.min_num_pts
			xz = tf.math.reduce_sum( (xpos - mu[:,:,0])*(zpos - mu[:,:,2]), axis = 1)/self.min_num_pts
			yz = tf.math.reduce_sum( (ypos - mu[:,:,1])*(zpos - mu[:,:,2]), axis = 1)/self.min_num_pts

			sigma = tf.Variable([xx, xy, xz,
								 xy, yy, yz,
								 xz, yz, zz]) 
			sigma = tf.reshape(tf.transpose(sigma), (tf.shape(sigma)[1] ,3,3))

			return(mu[:,0,:], sigma)


	def get_points_inside(self, cloud, cells):
		""" returns ragged tensor containing the indices of points in <cloud> inside each cell in <cells>"""

		thetamin = -np.pi
		thetamax = np.pi #-  2*np.pi/self.fid_theta
		phimin =  3*np.pi/8
		# phimax = 5*np.pi/8 #was this
		phimax = 7*np.pi/8 #why is this not the same as in <grid_spherical>????


		edges_phi = tf.linspace(phimin, phimax, self.fid_phi) #was this for regular cells
		# edges_phi, _ = tf.unique(self.grid[:,2])
		bins_phi = tfp.stats.find_bins(cloud[:,2], edges_phi)

		#works for regular voxels only--------------------------
		edges_r, _ = tf.unique(self.grid[:,0]) 
		bins_r = tfp.stats.find_bins(cloud[:,0], edges_r) 
		#-------------------------------------------------------

		#for extended radius brim voxels------------------------

		# # #TODO - need to modify the code for "get_occupied()"

		# # #temporarily half the radius measurement of every point with a phi value that puts it in the lower n "brim" bins to keep indexing working
		# temp_r = (cloud[:,0] - 3)*(1 - (cloud[:,2]//edges_phi[-3])/2) + 3
		# # print(temp_r[:,None])
		# # print(cloud[:,1:])
		# test = tf.concat((temp_r[:,None], cloud[:,1:]), axis = 1)
		# # print(test)
		# # print(edges_phi[-3])
		# # print((cloud[:,2]//edges_phi[-3]))
		# self.disp.append(Points(self.s2c(test).numpy(), c = 'green', r = 5 ))
		# edges_r, _ = tf.unique(self.grid[:,0])
		# bins_r = tfp.stats.find_bins(temp_r, edges_r) 
		#-------------------------------------------------------

		edges_theta = tf.linspace(thetamin, thetamax, self.fid_theta + 1)
		# edges_theta, _ = tf.unique(self.grid[:,1])
		bins_theta = tfp.stats.find_bins(cloud[:,1], edges_theta)
		# print("edges_theta", edges_theta)
		# print("bins_theta", bins_theta)		

		#cell index for every point in cloud
		cell_idx = tf.cast( bins_theta*(self.fid_phi-1) + bins_phi + bins_r*self.fid_theta*(self.fid_phi-1), tf.int32) #works for regular cells

		pts_in_c = tf.where(cell_idx == cells)
		
		numPtsPerCell = tf.math.bincount(tf.cast(pts_in_c[:,0], tf.int32))

		pts_in_c = tf.RaggedTensor.from_value_rowids(pts_in_c[:,1], pts_in_c[:,0]) 
		# pts_in_c = tf.RaggedTensor.from_value_rowids(pts_in_c[:,1], pts_in_c[:,0], nrows = tf.shape(cells)[0].numpy())
	
		return(pts_in_c, numPtsPerCell)


	def get_occupied(self, tophat = 0):
		""" returns idx of all voxels that occupy the line of sight closest to the observer """

		st = time.time()

		#attempt #2:------------------------------------------------------------------------------
		#bin points by spike
		thetamin = -np.pi
		thetamax = np.pi #-  2*np.pi/self.fid_theta
		phimin =  3*np.pi/8
		# phimax = 5*np.pi/8 #was this
		phimax = 7*np.pi/8 #why is this not the same as in <grid_spherical>????

		edges_phi = tf.linspace(phimin, phimax, self.fid_phi)
		edges_theta = tf.linspace(thetamin, thetamax, self.fid_theta + 1)

		cloud = self.cloud1_tensor_spherical
		if tophat == 1:
			temp_r = (cloud[:,0] - 3)*(1 - (cloud[:,2]//edges_phi[-3])/2) + 3
			cloud = tf.concat((temp_r[:,None], cloud[:,1:]), axis = 1)

		bins_theta = tfp.stats.find_bins(cloud[:,1], edges_theta)
		# print(bins_theta)
		bins_phi = tfp.stats.find_bins(cloud[:,2], edges_phi)
		# print(bins_phi)

		#combine bins_theta and bins_phi to get spike bins
		bins_spike = tf.cast(bins_theta*(self.fid_phi-1) + bins_phi, tf.int32)
		# print(tf.unique(bins_spike))
		# print("bins_spike", bins_spike)
		# self.draw_cell(tf.cast(bins_spike, tf.int32))

		#save which spike each point is in to ICET object for further analysis
		self.bins_spike = bins_spike

		#find min point in each occupied spike
		occupied_spikes, idxs = tf.unique(bins_spike)
		# print("occupied_spikes:", occupied_spikes)
		# print(idxs)

		temp =  tf.where(bins_spike == occupied_spikes[:,None]) #TODO- there has to be a better way to do this... 
		# print(temp)
		rag = tf.RaggedTensor.from_value_rowids(temp[:,1], temp[:,0])
		# print(rag)

		idx_by_rag = tf.gather(cloud[:,0], rag)
		# print(idx_by_rag)

		min_per_spike = tf.math.reduce_min(idx_by_rag, axis = 1)
		# print("min_per_spike:", min_per_spike)

		#get closest shell for each point in min_per_spike
		# print(tf.unique(self.grid[:,0]))
		radii, _ = tf.unique(self.grid[:,0])
		# print(radii)
		shell_idx = tf.math.reduce_sum(tf.cast(tf.greater(min_per_spike, radii[:,None] ), tf.int32), axis = 0) - 1
		# print(shell_idx)

		#find bin corresponding to the identified closeset points per cell
		occupied_cells = occupied_spikes + shell_idx*self.fid_theta*(self.fid_phi -1)
		# print("occupied_cells:", occupied_cells)

		# print("took", time.time() - st, "s to find occupied cells")
		return(occupied_cells)


	def draw_cell(self, corners, bad = False):
		""" draws cell provided by corners tensor"""

		# corners = self.get_corners(idx)
		# print(corners)

		if bad == False:
			for i in range(tf.shape(corners)[0]):
				p1, p2, p3, p4, p5, p6, p7, p8 = self.s2c(corners[i]).numpy()

				lineWidth = 1
				c1 = 'black'

				# arc1 = shapes.Arc(center = [0,0,0], point1 = p1, point2 = p2, c = 'red')	
				arc1 = shapes.Line(p1, p2, c = c1, lw = lineWidth) #debug		
				self.disp.append(arc1)
				# arc2 = shapes.Arc(center = [0,0,0], point1 = p3, point2 = p4, c = 'red')
				arc2 = shapes.Line(p3, p4, c = c1, lw = lineWidth) #debug
				self.disp.append(arc2)
				line1 = shapes.Line(p1, p3, c = c1, lw = lineWidth)
				self.disp.append(line1)
				line2 = shapes.Line(p2, p4, c = c1, lw = lineWidth) #problem here
				self.disp.append(line2)

				# arc3 = shapes.Arc(center = [0,0,0], point1 = p5, point2 = p6, c = 'red')		
				arc3 = shapes.Line(p5, p6, c = c1, lw = lineWidth) #debug
				self.disp.append(arc3)
				# arc4 = shapes.Arc(center = [0,0,0], point1 = p7, point2 = p8, c = 'red')
				arc4 = shapes.Line(p7, p8, c = c1, lw = lineWidth) #debug
				self.disp.append(arc4)
				line3 = shapes.Line(p5, p7, c = c1, lw = lineWidth)
				self.disp.append(line3)
				line4 = shapes.Line(p6, p8, c = c1, lw = lineWidth)
				self.disp.append(line4)

				self.disp.append(shapes.Line(p1,p5, c = c1, lw = lineWidth))
				self.disp.append(shapes.Line(p2,p6, c = c1, lw = lineWidth))
				self.disp.append(shapes.Line(p3,p7, c = c1, lw = lineWidth))
				self.disp.append(shapes.Line(p4,p8, c = c1, lw = lineWidth))

		if bad == True:
			#identified as containing moving objects
			for i in range(tf.shape(corners)[0]):
				p1, p2, p3, p4, p5, p6, p7, p8 = self.s2c(corners[i]).numpy()
				thicc = 3

				# arc1 = shapes.Arc(center = [0,0,0], point1 = p1, point2 = p2, c = 'yellow')	
				# arc1.lineWidth(thicc)
				arc1 = shapes.Line(p1, p2, c = 'yellow', lw = thicc) #debug
				self.disp.append(arc1)
				# arc2 = shapes.Arc(center = [0,0,0], point1 = p3, point2 = p4, c = 'yellow')
				# arc2.lineWidth(thicc)
				arc2 = shapes.Line(p3, p4, c = 'yellow', lw = thicc) #debug
				self.disp.append(arc2)
				line1 = shapes.Line(p1, p3, c = 'yellow', lw = thicc)
				self.disp.append(line1)
				line2 = shapes.Line(p2, p4, c = 'yellow', lw = thicc) #problem here
				self.disp.append(line2)

				# arc3 = shapes.Arc(center = [0,0,0], point1 = p5, point2 = p6, c = 'yellow')
				# arc3.lineWidth(thicc)
				arc3 = shapes.Line(p5, p6, c = 'yellow', lw = thicc) #debug			
				self.disp.append(arc3)
				# arc4 = shapes.Arc(center = [0,0,0], point1 = p7, point2 = p8, c = 'yellow')
				# arc4.lineWidth(thicc)
				arc4 = shapes.Line(p7, p8, c = 'yellow', lw = thicc) #debug
				self.disp.append(arc4)
				line3 = shapes.Line(p5, p7, c = 'yellow', lw = thicc)
				self.disp.append(line3)
				line4 = shapes.Line(p6, p8, c = 'yellow', lw = thicc)
				self.disp.append(line4)

				self.disp.append(shapes.Line(p1,p5,c = 'yellow', lw = thicc))
				self.disp.append(shapes.Line(p2,p6,c = 'yellow', lw = thicc))
				self.disp.append(shapes.Line(p3,p7,c = 'yellow', lw = thicc))
				self.disp.append(shapes.Line(p4,p8,c = 'yellow', lw = thicc))

		if bad == 2:
			#identified as perspective shift
			for i in range(tf.shape(corners)[0]):
				p1, p2, p3, p4, p5, p6, p7, p8 = self.s2c(corners[i]).numpy()
				thicc = 3

				# arc1 = shapes.Arc(center = [0,0,0], point1 = p1, point2 = p2, c = 'yellow')	
				# arc1.lineWidth(thicc)
				arc1 = shapes.Line(p1, p2, c = 'purple', lw = thicc) #debug
				self.disp.append(arc1)
				# arc2 = shapes.Arc(center = [0,0,0], point1 = p3, point2 = p4, c = 'yellow')
				# arc2.lineWidth(thicc)
				arc2 = shapes.Line(p3, p4, c = 'purple', lw = thicc) #debug
				self.disp.append(arc2)
				line1 = shapes.Line(p1, p3, c = 'purple', lw = thicc)
				self.disp.append(line1)
				line2 = shapes.Line(p2, p4, c = 'purple', lw = thicc) #problem here
				self.disp.append(line2)

				# arc3 = shapes.Arc(center = [0,0,0], point1 = p5, point2 = p6, c = 'yellow')
				# arc3.lineWidth(thicc)
				arc3 = shapes.Line(p5, p6, c = 'purple', lw = thicc) #debug			
				self.disp.append(arc3)
				# arc4 = shapes.Arc(center = [0,0,0], point1 = p7, point2 = p8, c = 'yellow')
				# arc4.lineWidth(thicc)
				arc4 = shapes.Line(p7, p8, c = 'purple', lw = thicc) #debug
				self.disp.append(arc4)
				line3 = shapes.Line(p5, p7, c = 'purple', lw = thicc)
				self.disp.append(line3)
				line4 = shapes.Line(p6, p8, c = 'purple', lw = thicc)
				self.disp.append(line4)

				self.disp.append(shapes.Line(p1,p5,c = 'purple', lw = thicc))
				self.disp.append(shapes.Line(p2,p6,c = 'purple', lw = thicc))
				self.disp.append(shapes.Line(p3,p7,c = 'purple', lw = thicc))
				self.disp.append(shapes.Line(p4,p8,c = 'purple', lw = thicc))


	def grid_spherical(self, draw = False):
		""" constructs grid in spherical coordinates """

		self.fid_r = self.fid  #waaayyy too many but keeping this for now
		self.fid_theta = self.fid  #number of subdivisions in horizontal directin
		self.fid_phi = self.fid_theta // 3

		thetamin = -np.pi 
		thetamax = np.pi - 2*np.pi/self.fid_theta #different from limits in main()
		phimin =  3*np.pi/8
		phimax = 7*np.pi/8

		a = tf.cast(tf.linspace(0,self.fid_r-1, self.fid_r)[:,None], tf.float32)
		b = tf.linspace(thetamin, thetamax, self.fid_theta)[:,None]
		c = tf.linspace(phimin, phimax, self.fid_phi)[:,None]

		ansb = tf.tile(tf.reshape(tf.tile(b, [1,self.fid_phi]), [-1,1] ), [(self.fid_r), 1])
		ansc = tf.tile(c, [self.fid_theta*self.fid_r, 1])
		#need to iteratively adjust spacing of radial positions to make cells roughly cubic

		nshell = self.fid_theta*(self.fid_phi) #number of grid cells per shell
		r_last = self.min_cell_distance #radis of line from observer to previous shell
		temp = np.ones([tf.shape(ansc)[0], 1])*self.min_cell_distance
		for i in range(1,self.fid_r):
			r_new = r_last*(1 + (np.arctan(2*np.pi/self.fid_theta))) #(cubic)
			# r_new = (r_last*(1 + (np.arctan(2*np.pi/self.fid_theta)))- 3)* 1.25 + 3 #(stretched)
			temp[(i*nshell):((i+1)*nshell+1),0] = r_new
			r_last = r_new
		ansa = tf.convert_to_tensor(temp, tf.float32)

		self.grid = tf.cast(tf.squeeze(tf.transpose(tf.Variable([ansa,ansb,ansc]))), tf.float32)
		# print(self.grid)

		# #test- double grid distance for bottom ring in each shell ~~~~~~~~~~
		#doing this messes with how we find points inside each cell

		#using only TF (unnecessarily difficult)
		# # print("lowest vertical angs", phimin )
		# indices = tf.where(self.grid[:,2] == phimin )
		# # print("indices", indices)
		# updates = tf.ones(tf.shape(indices))
		# # print("updates", tf.shape(updates))
		# # shape = tf.cast(tf.shape(self.grid[:,2]), tf.int64)
		# shape = tf.cast(tf.constant([self.fid_theta*self.fid_r*self.fid_phi, 1]), tf.int64)
		# # print("shape", shape)

		# mask = tf.scatter_nd(indices, updates, shape)
		# # print(mask)
		# # print(tf.shape(mask))
		# # print(tf.shape(self.grid))


		# mask = tf.concat((mask, tf.zeros([tf.shape(mask)[0], 2])), 1)
		# # mask = mask + tf.ones(tf.shape(mask))
		# print(mask)
		# # self.grid = self.grid + mask

		# ### using np (sligthly slower but fine for prototype) ------
		# # self.grid_simple = self.grid #save copy of grid with thee new radial spacing for bottom ring
		# n_ground_cells = 3
		# eps = 1e-4 #small value to give bin a discrete size

		# gnp = self.grid.numpy()
		# idx_bottom = np.where(gnp[:,2] >= c[-n_ground_cells].numpy())
		# # print(idx)
		# #expand radius of bottom two rings rows 
		# gnp[idx_bottom,0] = 2*gnp[idx_bottom,0]-3 

		# #set elevation angle of ring above bottom to be the same as below to avoid sloped voxels
		# # idx_next_up = np.where(gnp[:,2] == c[-(n_ground_cells + 1)].numpy())
		# # gnp[idx_next_up, 2] = gnp[np.where(gnp[:,2] == c[-n_ground_cells].numpy()), 2] + eps
		# # print(c) #grid of phi
		# # print(idx_next_up)

		# #raise elevation of "ground cells" by 1 to remove slope
		# # gnp[idx_bottom, 2] = gnp[np.asarray(idx_bottom) - 1, 2] + eps

		# self.grid_tophat = tf.convert_to_tensor(gnp)
		# # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		if draw == True:
			gp = self.s2c(self.grid.numpy())
			# print(gp)
			p = Points(gp, c = [0.3,0.8,0.3], r = 5)
			self.disp.append(p)

	def c2s(self, pts):
		""" converts points from cartesian coordinates to spherical coordinates """
		r = tf.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
		phi = tf.math.acos(pts[:,2]/r)
		theta = tf.math.atan2(pts[:,1], pts[:,0])

		out = tf.transpose(tf.Variable([r, theta, phi]))
		return(out)

	def s2c(self, pts):
		"""converts spherical -> cartesian"""

		x = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.cos(pts[:,1])
		y = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.sin(pts[:,1]) 
		z = pts[:,0]*tf.math.cos(pts[:,2])

		out = tf.transpose(tf.Variable([x, y, z]))
		# out = tf.Variable([x, y, z])
		return(out)

	def draw_cloud(self, points, pc = 1):

		if pc == 1:
			color = [0.8, 0.5, 0.5]
		if pc == 2:
			color = [0.5, 0.5, 0.8]
		if pc == 3:
			color = [0.5, 0.8, 0.5]
		
		c = Points(points, c = color, r = 2.5, alpha = 1.) #r = 4
		self.disp.append(c)

	def draw_car(self):
		# (used for making presentation graphics)
		fname = "honda.stl"
		# car = Mesh(fname).c("gray").rotate(90, axis = (0,0,1)).addShadow(z=-1.85) #old vedo
		car = Mesh(fname).c("gray").rotate(90, axis = (0,0,1))
		car.pos(1.4,1,-1.72)
		car.addShadow(plane = 'z', point = -1.85, c=(0.5, 0.5, 0.5))
		self.disp.append(car)