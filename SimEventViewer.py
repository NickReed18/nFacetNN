import matplotlib
import tk
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
import pandas as pd
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator,FormatStrFormatter,MaxNLocator)
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import median_absolute_deviation
import os
from math import isinf
import itertools
import ndap
from scipy.optimize import curve_fit

#NEED TO RESTRUCTURE SOMEWHAT
class SimEventViewer():
    """Event viewer for simulated events

    Generates a 3D view of the detector and displays detector events. 
    Currently written to work with root files directly, 
    will eventually be fully updated to work with h5 files.

    ATTRIBUTES
    --------------------
    metadata 
        metadata from root files

    initial    
        pandas DataFrame containing initial neutron data

    final 
        pandas DataFrame containing final neutron data 

    track
        pandas DataFrame containing neutron tracking data

    detresponse
        pandas DataFrame containing Geant4 SD responses

    detloc
        location of the detector in mm, array of shape (3,)

    cubesize
        size of detector cubes in mm

    curr_pos
        index for scrolling through scatters for event display

    
    METHODS
    -------------------
    PreProcess(self)
        Preprocesses data for event viewing

    InteractiveEventViewer(self,rot=180,batchsize=1,batch=False,rot_axes=(0,1))
        Plots the interactive event viewer

    TEScatterDist(self,cut_therm=True)
        Plots the 2d histogram of time between scatter and energy loss in scatter
    
    CapEfficiency(self)
        Calculates the capture efficiency, but only for root file 
        inputs (as h5 file inputs are curated to only be events in the detector already)
    
    ScatterInitEDist(self)
        Plots the 2d histogram of initial neutron energy and number of scatters in the detector

    CapTimeDist(self,log=False)
        Plots histogram of neutron capture times, to demonstrate the exponential full off


    """


    def __init__(self,rootfile,source,detloc=np.array([0,-1627,1500]),cubesize=50.888):
        """
        Defines initial quantities, does preprocessing of data e.g. reconstruction of physical distance from cubes

        PARAMETERS
        -----------
        rootfile : str
            The path of the desired file
        
        detloc : ndarray
            The location of the detector, in mm

        cubesize : float
            The edge length of the detector cubes, in mm

        """
        if not isinstance(source, str):
            raise TypeError("Source must be passed as a string")
        self.processed=False
        self.source=source
        self.cubedist=None
        if rootfile[-2:]=='h5':
            h5=True
        else:
            h5=False
        if not h5:
            file=uproot.open(rootfile)
            self.metadata=file['T0;1'].arrays()
            self.initial=file['T1;1'].arrays()
            self.final=file['T2;1'].arrays()
            self.track=file['T3;1'].arrays()
            self.detresponse=file['T5;1'].arrays()

        if h5:
            file=pd.read_hdf(rootfile)
            file.reset_index(inplace=True)
            file.drop('index',axis=1,inplace=True)
            self.initial=file.loc[:,['pdgid','energy','posx','posy','posz','momx','momy','momz']]
            self.final=file.loc[:,['DepositEnergy','LiCaptureEnergy','PvtCaptureEnergy','CapPosX',
                                   'CapPosY','CapPosZ','ThermalPosX','ThermalPosY','ThermalPosZ',
                                   'CapE','LiCapture','PVTCapture','CapCube','Detected','Generated1',
                                   'Generated2','WLS','ModerationTime','ThermalTime','CaptureTime',
                                   'ThermalRadius','ModDistance','CapDistance','NScatter','LiTotCross',
                                   'LiFastCross','LiThermalCross','FurthestDistance','HDPEReflection']]
            self.track=file.loc[:,['StepTime','Energy','XPosition','YPosition','ZPosition','StepLength','EnergyChange']]
            self.detresponse=file.loc[:,['cubeid','cubex','cubey','cubez','pdg','edep_tot','edep_pvt',
                                         'edep_zns','time_last','tracklength']]
        self.detloc=detloc
        self.cubesize=cubesize
        self.curr_pos=0
        self.h5 = h5

        if not self.h5:
            self.xp=self.track[b'XPosition']
            self.yp=self.track[b'YPosition']
            self.zp=self.track[b'ZPosition']
            self.edep=np.where(self.detresponse[b'edep_zns'].sum()>1)[0]

            times=self.track[b'StepTime']
            starts1=[]
            starts2=[]
            for i in range(len(times)):
                if times[i].size==0:
                    starts1.append(i)

            for i in range(len(times)):
                if i==len(times)-1:
                    continue
                if np.logical_and(times[i].size<1,times[i+1].size>0):
                    starts2.append(i)
            bad=np.where(~np.isin(starts1,starts2))[0]
            self.starts1=starts1
            self.starts2=starts2
            self.bad=bad
            starts=np.delete(starts1,bad[1::2])
            self.ends=np.roll(starts-1,-1)

            self.xmin,self.xmax=self.detloc[0]-2*self.cubesize,self.detloc[0]+2*self.cubesize
            self.ymin,self.ymax=self.detloc[1]-2*self.cubesize,self.detloc[1]+2*self.cubesize
            self.zmin,self.zmax=self.detloc[2]-2*self.cubesize,self.detloc[2]+2*self.cubesize

            xcomp,ycomp,zcomp=[],[],[]
            for arr in self.xp:
                for element in arr:
                    xcomp.append(element)
            for arr in self.yp:
                for element in arr:
                    ycomp.append(element)
            for arr in self.zp:
                for element in arr:
                    zcomp.append(element)
            

            xin=np.where(np.logical_and(xcomp[i]>self.xmin,xcomp[i]<self.xmax))[0]
            yin=np.where(np.logical_and(ycomp[i]>self.ymin,ycomp[i]<self.ymax))[0]
            zin=np.where(np.logical_and(zcomp[i]>self.zmin,zcomp[i]<self.zmax))[0]
            self.track_indet=np.intersect1d(np.intersect1d(xin,yin),zin)
            vp=[]
            for i in range(len(self.edep)):
                x=np.array(self.xp[self.ends[self.edep]][i])
                y=np.array(self.yp[self.ends[self.edep]][i])
                z=np.array(self.zp[self.ends[self.edep]][i])
            
                x=np.append(x,self.final[b'CapPosX'][self.edep][i])
                x-=(self.detloc[0]-2*self.cubesize)
                x/=self.cubesize

                y=np.append(y,self.final[b'CapPosY'][self.edep][i])
                y-=(self.detloc[1]-2*self.cubesize)
                y/=self.cubesize

                z=np.append(z,self.final[b'CapPosZ'][self.edep][i])
                z-=(self.detloc[2]-2*self.cubesize)
                z/=self.cubesize        

                point=np.vstack((x,y,z))
                vp.append(point)
            self.visible_tracks=vp
            self.processed=True
            return 

        else:
            edep=[]
            e=self.detresponse['edep_zns'].array
            for i in range(len(e)):
                if e[i].sum()>1:
                    edep.append(i)
            self.edep=edep
            self.xp=self.track['XPosition'].array[self.edep]
            self.yp=self.track['YPosition'].array[self.edep]
            self.zp=self.track['ZPosition'].array[self.edep]

            self.xmin,self.xmax=self.detloc[0]-2*self.cubesize,self.detloc[0]+2*self.cubesize
            self.ymin,self.ymax=self.detloc[1]-2*self.cubesize,self.detloc[1]+2*self.cubesize
            self.zmin,self.zmax=self.detloc[2]-2*self.cubesize,self.detloc[2]+2*self.cubesize

            xcomp,ycomp,zcomp=[],[],[]
            for arr in self.xp:
                for element in arr:
                    xcomp.append(element)
            for arr in self.yp:
                for element in arr:
                    ycomp.append(element)
            for arr in self.zp:
                for element in arr:
                    zcomp.append(element)
            

            xin=np.where(np.logical_and(xcomp>self.xmin,xcomp<self.xmax))[0]
            yin=np.where(np.logical_and(ycomp>self.ymin,ycomp<self.ymax))[0]
            zin=np.where(np.logical_and(zcomp>self.zmin,zcomp<self.zmax))[0]
            self.track_indet=np.intersect1d(np.intersect1d(xin,yin),zin)



            vp=[]
            for i in range(len(self.xp)):
                x=self.xp[i]
                y=self.yp[i]
                z=self.zp[i]

                x=np.append(x,self.final['CapPosX'].array[i])
                x-=(self.detloc[0]-2*self.cubesize)
                x/=self.cubesize

                y=np.append(y,self.final['CapPosY'].array[i])
                y-=(self.detloc[1]-2*self.cubesize)
                y/=self.cubesize

                z=np.append(z,self.final['CapPosZ'].array[i])
                z-=(self.detloc[2]-2*self.cubesize)
                z/=self.cubesize

                point=np.vstack((x,y,z))
                vp.append(point)
            self.visible_tracks=vp
            self.processed=True
            return        
    
    def InteractiveEventViewer(self,rot=180,batchsize=1,batch=False,rot_axes=(0,1)):
        """
        Generates the interactive event viewer

        The viewer is a 3D view of the detector layout. 
        Detectable events can be scrolled through with arrow key presses, 
        and events can be displayed in batches.

        PARAMETERS
        -----------
        rot : int
            Rotation of the detector from normal orientation. 
            Defaults to 180 degrees. Must be a multiple of 90 degrees 
            or a ValueError will be raised.

        batchsize : int
            Number of events to display at once

        batch : bool
            Whether or not you want to display a batch of events. 

        rot_axes : tuple
            Which axes to rotate, first towards the second (e.g. (0, 1) causes an x-y rotation)


        RAISES
        --------
        ValueError
            If the rot variable is not a multiple of 90 degrees
        """

        if rot%90!=0:
            raise ValueError("Rotation must be a multiple of 90 degrees")

        self.cap_off=False
        self.track_off=False
        self.detect_off=False
        self.curr_pos=0
        subject=np.ones((4,4,4))
        sc=(self.cubesize-50)/50
        plotter=ndap.NDArrayPlotter(subject,spacing=("even",sc,sc,sc))
        plotter.set_alpha(0.05)

        energies=self.track['EnergyChange'][self.edep].explode().iloc[self.track_indet].array
        detectable=[]
        for i in range(len(self.visible_tracks)):
            detectable.append(np.where(energies[i]<=-4.8e5)[0])
        detectable=np.array(detectable)

        if self.h5:
            cubex=self.detresponse['cubex'].array[self.edep]
            cubey=self.detresponse['cubey'].array[self.edep]
            cubez=self.detresponse['cubez'].array[self.edep]

        else:
            cubex=self.detresponse[b'cubex'][self.edep]
            cubey=self.detresponse[b'cubey'][self.edep]
            cubez=self.detresponse[b'cubez'][self.edep]

        cmap=cm.viridis
        scattered=[]
        for i in range(len(cubex)):
            col=np.full((4,4,4),0.)
            for j in range(len(cubex[i])):
                col[cubex[i][j]-1,
                    cubey[i][j]-1,
                    cubez[i][j]-1]+=1
            cols=np.rot90(col,rot/90,axes=rot_axes)
            scattered.append(cols)
            

        nbatches=int(len(self.visible_tracks)/batchsize)
        points=np.arange(0,len(self.visible_tracks))
        self.batches=np.split(points[:points.size-points.size%batchsize],nbatches)
        self.batches.append(points[points.size-points.size%batchsize:])

        def key_event(e):

            if e.key=="right":
                self.curr_pos+=1
                if batch:
                    self.curr_pos %= len(self.batches)
                else:
                    self.curr_pos %=len(self.visible_tracks)

                ax.cla()
                if batch:
                    display=np.zeros((4,4,4))
                    for i in self.batches[self.curr_pos]:
                        display=display+scattered[i]
                    norm = colors.Normalize(vmin=np.min(display),vmax=np.max(display))
                    plotter.colors=cmap(norm(display))
                    alph=norm(display)*0.25
                    alph+=0.05
                    plotter.alphas=alph
                else:
                    display=scattered[self.curr_pos]
                    norm=colors.Normalize(vmin=np.min(display),vmax=np.max(display))
                    plotter.colors=cmap(norm(display))
                    alph=norm(display)*0.25
                    alph+=0.05
                    plotter.alphas=alph
                plotter.render(azim=-45,elev=35,ax=ax)
                self.cbar.set_clim(vmin=0,vmax=np.max(display))
                self.cbar.draw_all()
                self.tracks=[]
                self.caps=[]
                self.detects=[]
                if batch:
                    for i in self.batches[self.curr_pos]:
                        self.tracks.append(ax.plot(self.visible_tracks[i][0],
                                              self.visible_tracks[i][1],
                                              self.visible_tracks[i][2],
                                              c='b'))
                        self.caps.append(ax.scatter(self.visible_tracks[i][0][-1],
                                               self.visible_tracks[i][1][-1],
                                               self.visible_tracks[i][2][-1],c='xkcd:cyan',s=15))
                        self.detects.append(ax.scatter(self.visible_tracks[i][0][detectable[i]],
                                                       self.visible_tracks[i][1][detectable[i]],
                                                       self.visible_tracks[i][2][detectable[i]],c='r',s=15))
                    
                else:
                    self.tracks.append(ax.plot(self.visible_tracks[self.curr_pos][0],
                                          self.visible_tracks[self.curr_pos][1],
                                          self.visible_tracks[self.curr_pos][2],
                                          c='b'))

                    self.caps.append(ax.scatter(self.visible_tracks[self.curr_pos][0][-1],
                                           self.visible_tracks[self.curr_pos][1][-1],
                                           self.visible_tracks[self.curr_pos][2][-1],
                                           c='xkcd:cyan',s=15))
                    self.detects.append(ax.scatter(self.visible_tracks[self.curr_pos][0][detectable[self.curr_pos]],
                                                   self.visible_tracks[self.curr_pos][1][detectable[self.curr_pos]],
                                                   self.visible_tracks[self.curr_pos][2][detectable[self.curr_pos]],c='r',s=15))
                if self.track_off:
                    for track in self.tracks:
                        for point in track:
                            point.set_visible(not point.get_visible())
                if self.cap_off:
                    for cap in self.caps:
                        cap.set_visible(not cap.get_visible())
                if self.detect_off:
                    for det in self.detects:
                        det.set_visible(not det.get_visible())
                fig.canvas.draw()

            elif e.key=="left":
                self.curr_pos-=1
                if batch:
                    self.curr_pos %= len(self.batches)
                else:
                    self.curr_pos %=len(self.visible_tracks)

                ax.cla()
                if batch:
                    display=np.zeros((4,4,4))
                    for i in self.batches[self.curr_pos]:
                        display=display+scattered[i]
                    norm=colors.Normalize(vmin=np.min(display),vmax=np.max(display))
                    plotter.colors=cmap(norm(display))
                    alph=norm(display)*0.25
                    alph+=0.05
                    plotter.alphas=alph
                else:
                    display=scattered[self.curr_pos]
                    norm=colors.Normalize(vmin=np.min(display),vmax=np.max(display))
                    plotter.colors=cmap(norm(display))
                    alph=norm(display)*0.25
                    alph+=0.05
                    plotter.alphas=alph
                plotter.render(azim=-45,elev=35,ax=ax)
                self.cbar.set_clim(vmin=0,vmax=np.max(display))
                self.cbar.draw_all()
                self.tracks=[]
                self.caps=[]
                self.detects=[]
                if batch:
                    for i in self.batches[self.curr_pos]:
                        self.tracks.append(ax.plot(self.visible_tracks[i][0],
                                              self.visible_tracks[i][1],
                                              self.visible_tracks[i][2],
                                              c='b'))
                        self.caps.append(ax.scatter(self.visible_tracks[i][0][-1],
                                               self.visible_tracks[i][1][-1],
                                               self.visible_tracks[i][2][-1],c='xkcd:cyan',s=15))
                        self.detects.append(ax.scatter(self.visible_tracks[i][0][detectable[i]],
                                                       self.visible_tracks[i][1][detectable[i]],
                                                       self.visible_tracks[i][2][detectable[i]],c='r',s=15))
                        

                else:
                    self.tracks.append(ax.plot(self.visible_tracks[self.curr_pos][0],
                                          self.visible_tracks[self.curr_pos][1],
                                          self.visible_tracks[self.curr_pos][2],
                                          c='b'))
                    self.caps.append(ax.scatter(self.visible_tracks[self.curr_pos][0][-1],
                                           self.visible_tracks[self.curr_pos][1][-1],
                                           self.visible_tracks[self.curr_pos][2][-1],c='xkcd:cyan',s=15))
                    self.detects.append(ax.scatter(self.visible_tracks[self.curr_pos][0][detectable[self.curr_pos]],
                                                   self.visible_tracks[self.curr_pos][1][detectable[self.curr_pos]],
                                                   self.visible_tracks[self.curr_pos][2][detectable[self.curr_pos]],c='r',s=15))

                if self.track_off:
                    for track in self.tracks:
                        for point in track:
                            point.set_visible(not point.get_visible())
                if self.cap_off:
                    for cap in self.caps:
                        cap.set_visible(not cap.get_visible())
                if self.detect_off:
                    for det in self.detects:
                        det.set_visible(not det.get_visible())
                fig.canvas.draw()
            elif e.key==".":
                for cap in self.caps:
                    cap.set_visible(not cap.get_visible())
                fig.canvas.draw()
                self.cap_off=~self.cap_off
                print(bool(self.cap_off))
            elif e.key=="/":
                for track in self.tracks:
                    for i in track:
                        i.set_visible(not i.get_visible())
                fig.canvas.draw()
                self.track_off=~self.track_off
            elif e.key==",":
                for det in self.detects:
                    det.set_visible(not det.get_visible())
                fig.canvas.draw()
                self.detect_off=~self.detect_off
            else:
                return
            

        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        fig.set_tight_layout(True)
        fig.suptitle(self.source,fontsize=24)

        fig.canvas.mpl_connect('key_press_event',key_event)
        if batch:
            display=np.zeros((4,4,4))
            for i in self.batches[0]:
                display=display+scattered[i]
            norm=colors.Normalize(vmin=np.min(display),vmax=np.max(display))
            plotter.colors=cmap(norm(display))
            alph=norm(display)*0.25
            alph+=0.05
            plotter.alphas=alph
            sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
            sm.set_array(display)
        else:
            norm=colors.Normalize(vmin=np.min(scattered[0]),vmax=np.max(scattered[0]))
            plotter.colors=cmap(norm(scattered[0]))
            alph=norm(scattered[0])*0.25
            alph+=0.05
            plotter.alphas=alph
            sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
            sm.set_array(scattered[0])

        plotter.render(azim=-45,elev=25,ax=ax)
        self.cbar=plt.colorbar(sm,ax=ax)
        self.cbar.set_label('Number of detected interactions',rotation=270,fontsize=20,labelpad=30)
        self.tracks=[]
        self.caps=[]
        self.detects=[]
        if batch:
            for i in range(batchsize):
                self.tracks.append(ax.plot(self.visible_tracks[i][0],
                        self.visible_tracks[i][1],
                        self.visible_tracks[i][2],
                        c='b'))
                self.caps.append(ax.scatter(self.visible_tracks[i][0][-1],
                           self.visible_tracks[i][1][-1],
                           self.visible_tracks[i][2][-1],c='xkcd:cyan',s=15))
                self.detects.append(ax.scatter(self.visible_tracks[i][0][detectable[i]],
                                               self.visible_tracks[i][1][detectable[i]],
                                               self.visible_tracks[i][2][detectable[i]],c='r',s=15))
        else:
            self.tracks.append(ax.plot(self.visible_tracks[0][0],
                    self.visible_tracks[0][1],
                    self.visible_tracks[0][2],
                    c='b'))
            self.caps.append(ax.scatter(self.visible_tracks[0][0][-1],
                       self.visible_tracks[0][1][-1],
                       self.visible_tracks[0][2][-1],c='xkcd:cyan',s=15))
            self.detects.append(ax.scatter(self.visible_tracks[0][0][detectable[0]],
                                           self.visible_tracks[0][1][detectable[0]],
                                           self.visible_tracks[0][2][detectable[0]],c='r',s=15))
                                    

        plt.show()
        return 

    def TDScatterDist(self,plot_views=True):
        """
        Plots a 2D histogram of time between scatter vs distance travelled between scatter. 

        PARAMETERS
        ----------
        plot_views : bool
            Whether to plot the 1D histograms of time between scatter and distance travelled or not. 
            If true, these are both plotted alongside the 2D histogram.

        
        RAISES
        -------
        NotImplementedError
            If the data used is not a h5 file, as this is not implemented for a raw root file input
            All raw root file input code will be eventually removed as the processing approach for 
            simulation data has changed.
        """


        if not self.h5:
            raise NotImplementedError("Not implemented for raw simulation output files")
        if self.h5:
            time=self.track['StepTime'].array[self.edep]
            distance=self.track['StepLength'].array[self.edep]
            tcomp,dcomp=[],[]
            for i in range(len(distance)):
                t=np.roll(np.append(np.diff(time[i]),time[i][0]),1)
                d=distance[i]
                for j in range(len(t)):
                    tcomp.append(t[j])
                    dcomp.append(d[j])

            tcomp=np.array(tcomp)
            dcomp=np.array(dcomp)

            tplot=tcomp[self.track_indet]
            dplot=dcomp[self.track_indet]

            tbins=10**np.linspace(np.log10(min(tplot)),np.log10(max(tplot)),200)
            dbins=10**np.linspace(np.log10(min(dplot)),np.log10(max(dplot)),200)


            us=np.full(len(dbins),1000)
            ns=np.full(len(dbins),1000)

            if not plot_views:
                plt.figure(figsize=(19.2,10.8))
                plt.hist2d(tplot,dplot,bins=[tbins,dbins],norm=colors.LogNorm())
                plt.xscale('log')
                plt.yscale('log')
                cb=plt.colorbar()
                cb.ax.tick_params(labelsize=18)
                plt.xlabel('Time between scatter / ns',fontsize=20)
                plt.ylabel('Distance travelled between scatter / mm',fontsize=20)
                plt.plot(us,dbins,label='1 $\mu$s')
                plt.plot(ns,dbins,label='1 ns')
                plt.legend(loc='lower left')

                plt.show()
            if plot_views:
                fig=plt.figure(figsize=(19.2,10.8))
                gs=fig.add_gridspec(2,4)

                ax1=fig.add_subplot(gs[:,:2])
                hist=ax1.hist2d(tplot,dplot,bins=[tbins,dbins],norm=colors.LogNorm())
                ax1.set_xlabel('Time between scatter / ns',fontsize=20)
                ax1.set_ylabel('Distance travelled between scatter / mm',fontsize=20)
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                ax1.plot(us,dbins,label='1 $\mu$s')
                ax1.plot(ns,dbins,label='1 ns')
                ax1.legend(loc='lower left')
                cbar=plt.colorbar(hist[3],ax=ax1)
                cbar.ax.tick_params(labelsize=20)

                ax2=fig.add_subplot(gs[0,2:])
                ax2.hist(tplot,bins=tbins)
                ax2.set_xscale('log')
                ax2.set_xlabel('Time between scatter / ns',fontsize=20)

                ax3=fig.add_subplot(gs[1,2:])
                ax3.hist(dplot,bins=dbins)
                ax3.set_xscale('log')
                ax3.set_xlabel('Distance travelled between scatter / mm',fontsize=20)


                fig.suptitle(self.source,fontsize=24)
                plt.show()

    def TEScatterDist(self,cut_therm=True,plot_views=True,strict_cut=True):
        """
        Plots a 2d histogram showing the distribution of time between scatter against 
        the energy lost in a scatter.

        PARAMETERS
        ----------
        cut_therm : bool
            If True, cut the plot at the energy scale of thermal neutrons

        RAISES
        -------
        NotImplementedError
            If the data used is a raw root file, rather than the processed h5 file
            Any raw root file implementations will be phased out as the processing approach has changed.
        """
        if not self.h5:
            raise NotImplementedError("Not implemented for raw simulation output files")
        if self.h5:
            dist=self.track.loc[:,['EnergyChange','StepTime']]
            tcomp=[]
            time=dist['StepTime']
            for i in range(len(dist)):
                t=np.roll(np.append(np.diff(time[i]),time[i][0]),1)
                tcomp.append(t)
            dist['TimeDiff']=tcomp

            capx=self.final['CapPosX']
            capy=self.final['CapPosY']
            capz=self.final['CapPosZ']
            dist['capindet']=(capx>self.xmin)&(capx<self.xmax)&(capy>self.ymin)&(capy<self.ymax)&(capz>self.zmin)&(capz<self.zmax)

            e=dist.loc[:,['EnergyChange','capindet']].explode('EnergyChange')
            data=pd.concat((dist['TimeDiff'].explode(),e),axis=1)
            dist=None
            e=None
            data['EnergyChange']*=-1
            data['Visible']=data['EnergyChange']>=4.8e5

            data=data[data['EnergyChange']>0]



            if strict_cut:
                thresh=1
            else:
                thresh=0.005
            cut=data['EnergyChange']>thresh
            tplot=data['TimeDiff'][cut]
            eplot=data['EnergyChange'][cut]
            indet=data['capindet'][cut]
            visible=data['Visible'][cut]


            tbins=10**np.linspace(-5,5,200)
            ebins=10**np.linspace(np.log10(min(eplot)),np.log10(max(eplot)),200)

            us=np.full(200,1000)
            ns=np.full(200,1)
            thermal=np.full(200,0.025)
            detect=np.full(200,480000)
            
            if not plot_views:
                plt.figure(figsize=(19.2,10.8))
                plt.hist2d(tplot,eplot,bins=[tbins,ebins],norm=colors.LogNorm())
                plt.xscale('log')
                plt.yscale('log')
                plt.ylim(top=10e7)
                cb=plt.colorbar()
                cb.ax.tick_params(labelsize=18)
                plt.xlabel('Time between scatter / ns',fontsize=20)
                plt.ylabel('Energy loss in scatter / eV',fontsize=20)
                plt.fill_between(tbins,0,detect,alpha=0.3,label='Not detectable ($\\textless$ 480 keV)')
                plt.plot(us,ebins,label='1 $\mu$s')
                plt.plot(ns,ebins,label='1 ns')
                plt.plot(tbins,thermal,label='Thermal neutron energy (0.025 eV)')
                plt.legend(loc='lower left')
                if cut_therm:
                    plt.ylim(bottom=0.005)
                plt.show()
            if plot_views:
                fig=plt.figure(figsize=(19.2,10.8))
                gs=fig.add_gridspec(2,4)

                ax1=fig.add_subplot(gs[:,:2])
                hist=ax1.hist2d(tplot,eplot,bins=[tbins,ebins],norm=colors.LogNorm())
                ax1.set_xlabel('Time between scatter / ns',fontsize=20)
                ax1.set_ylabel('Energy loss in scatter / eV',fontsize=20)
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                ax1.set_ylim(top=10e7)
                ax1.fill_between(tbins,0,detect,alpha=0.3,label='Not detectable ($\\textless$ 480 keV electron equivalent)')
                ax1.plot(us,ebins,label='1 $\mu$s')
                ax1.plot(ns,ebins,label='1 ns')
                ax1.plot(tbins,thermal,label='Thermal neutron energy (0.025 eV)')
                ax1.legend(loc='lower left')
                cbar=plt.colorbar(hist[3],ax=ax1)
                cbar.ax.tick_params(labelsize=20)

                ax2=fig.add_subplot(gs[0,2:])
                ax2.hist(tplot,bins=tbins,color='darkblue',alpha=0.3,label='All events')
                ax2.hist(tplot[visible],bins=tbins,color='purple',alpha=0.8,label='Detectable events')
                ax2.hist(tplot[indet],bins=tbins,color='purple',alpha=0.5,label='Captured')
                ax2.set_xscale('log')
                ax2.set_xlabel('Time between scatter / ns',fontsize=20)
                ax2.set_ylabel('Frequency',fontsize=20)
                ax2.legend(loc='lower left')

                ax3=fig.add_subplot(gs[1,2:])
                ax3.hist(eplot,bins=ebins,color='darkblue',alpha=0.3,label='All events')
                ax3.hist(eplot[visible],bins=ebins,color='purple',alpha=0.8,label='Detectable events')
                ax3.hist(eplot[indet],bins=ebins,color='purple',alpha=0.5,label='Captured')
                ax3.set_xscale('log')
                ax3.set_xlabel('Energy loss in scatter / eV',fontsize=20)
                ax3.set_ylabel('Frequency',fontsize=20)


                fig.suptitle(self.source,fontsize=24)
                plt.show()

    def CapEfficiency(self):
        """
        Calculates the efficiency of capture for a root file. This is only done for a root file
        because the h5 files have already filtered events based on those that deposit sufficient 
        energy to be detected.

        RAISES
        ------
        Exception
            If the data has not been processed using the PreProcess method.
        """
        if not self.processed:
            raise Exception("Data not processed; please call PreProcess first")
        if self.h5:
            capx=self.final['CapPosX'].array
            capy=self.final['CapPosY'].array
            capz=self.final['CapPosZ'].array

            xin=np.where(np.logical_and(capx>self.xmin,capx<self.xmax))[0]
            yin=np.where(np.logical_and(capy>self.ymin,capy<self.ymax))[0]
            zin=np.where(np.logical_and(capz>self.zmin,capz<self.zmax))[0]

            captured=np.intersect1d(np.intersect1d(xin,yin),zin)
            perc=len(captured)/len(capx)
            perc*=100
            self.cap_eff=perc
            return perc
        else:
            x=self.xp[self.ends]
            y=self.yp[self.ends]
            z=self.zp[self.ends]
            capx=self.final[b'CapPosX']
            capy=self.final[b'CapPosY']
            capz=self.final[b'CapPosZ']
        
            entered=[]

            for i in range(len(x)):
                xin=(np.logical_and(x[i]>self.xmin,x[i]<self.xmax)).any()
                yin=(np.logical_and(y[i]>self.ymin,y[i]<self.ymax)).any()
                zin=(np.logical_and(z[i]>self.zmin,z[i]<self.zmax)).any()

                if xin and yin and zin:
                    entered.append(i)
        
            xin=np.where(np.logical_and(capx>self.xmin,capx<self.xmax))[0]
            yin=np.where(np.logical_and(capy>self.ymin,capy<self.ymax))[0]
            zin=np.where(np.logical_and(capz>self.zmin,capz<self.zmax))[0]

            captured=np.intersect1d(np.intersect1d(xin,yin),zin)

            perc=(len(captured)/len(entered))*100
            self.cap_eff=perc
            return perc

    def ScatterInitEDist(self,plot_views=True):
        """
        Plots a 2d histogram of the number of scatters in the detector against the 
        initial energy of the neutron.  

        PARAMETERS
        ----------
        plot_views : bool
            Whether to plot the 1D histograms of number of scatters and initial energy or not. 
            If True, these 1D histograms will be plotted alongside the 2D histogram.


        """
        if self.h5:
            x=self.track['XPosition'].array
            y=self.track['YPosition'].array
            z=self.track['ZPosition'].array
            e=self.initial['energy'].array
        else:
            x=self.xp
            y=self.yp
            z=self.zp
            e=self.initial[b'energy']
        
        nscatt=[]
        for i in range(len(x)):
            xin=np.where(np.logical_and(x[i]>self.xmin,x[i]<self.xmax))[0]
            yin=np.where(np.logical_and(y[i]>self.ymin,y[i]<self.ymax))[0]
            zin=np.where(np.logical_and(z[i]>self.zmin,z[i]<self.zmax))[0]

            inn=np.intersect1d(np.intersect1d(xin,yin),zin)
            nscatt.append(len(inn))

        eplot=[]
        for i in range(len(e)):
            eplot.append(e[i][0])
        if plot_views:
            fig=plt.figure(figsize=(19.2,10.8))
            gs=fig.add_gridspec(2,4)
            ax1=fig.add_subplot(gs[:,:2])
            hist=ax1.hist2d(np.array(eplot),np.array(nscatt),bins=50,norm=colors.LogNorm())
            ax1.set_xlabel('Initial energy / MeV',fontsize=20)
            ax1.set_ylabel('Number of scatters in detector',fontsize=20)
            cbar=plt.colorbar(hist[3],ax=ax1)
            cbar.ax.tick_params(labelsize=20)

            ax2=fig.add_subplot(gs[0,2:])
            ax2.hist(eplot,bins=50)
            ax2.set_xlabel('Initial energy / MeV',fontsize=20)

            ax3=fig.add_subplot(gs[1,2:])
            ax3.hist(nscatt,bins=50)
            ax3.set_xlabel('Number of scatters in detector',fontsize=20)

            fig.suptitle(self.source,fontsize=24)
        else:
            plt.figure(figsize=(19.2,10.8))
            plt.hist2d(eplot,nscatt,bins=50,norm=colors.LogNorm())
            plt.xlabel('Initial energy / MeV',fontsize=20)
            plt.ylabel('Number of scatters in detector',fontsize=20)
            cbar=plt.colorbar()
            cbar.ax.tick_params(labelsize=20)

        plt.show()

    def CapTimeDist(self,plotdir=None):
        """
        Plots a histogram of neutron capture times.

        PARAMETERS
        ----------
        plotdir : string
            Directory to save the figure to. If None, figure is not saved. 
        """
        if not self.processed:
            raise Exception("Data must be processed first")
        if self.h5:
            capt=self.final['CaptureTime'].array
        else:
            capt=self.final[b'CaptureTime']
        
        plt.figure(figsize=(19.2,10.8))
        plt.hist(capt,bins=100,range=(0,1000000))
        plt.xlabel("Capture time / ns",fontsize=20)
        plt.ylabel("Frequency",fontsize=20)
        plt.title("Capture time distribution",fontsize=24)
        plt.xlim(0,1e6)
        if plotdir!=None:
            plt.savefig(plotdir+"/"+self.source+"_capture_time_dist.png")
        plt.show() 
        return

    def AbsCapPosDist(self,plotdir=None):
        """
        Plots x, y and z absolute capture position histograms

        PARAMETERS
        -----------
        plotdir : str
            Directory to save the plot to. If None, the plot is not saved.

        """
        capx=self.final['CapPosX'].array
        capy=self.final['CapPosY'].array
        capz=self.final['CapPosZ'].array
        
        inx=np.where(np.logical_and(capx>self.xmin-5,capx<self.xmax+5))[0]
        iny=np.where(np.logical_and(capy>self.ymin-5,capy<self.ymax+5))[0]
        inz=np.where(np.logical_and(capz>self.zmin-5,capz<self.zmax+5))[0]
        capindet=np.intersect1d(np.intersect1d(inx,iny),inz)

        indetx=capx[capindet]
        indety=capy[capindet]
        indetz=capz[capindet]

        fig=plt.figure(figsize=(19.2,10.8))
        gs=fig.add_gridspec(3,5)
        ax1=fig.add_subplot(gs[0,:])
        xhist=ax1.hist(indetx,bins=250)
        ax1.set_xlabel('X capture position / mm',fontsize=20)
        ax1.set_ylabel('Frequency',fontsize=20)

        ax2=fig.add_subplot(gs[1,:])
        yhist=ax2.hist(indety,bins=250)
        ax2.set_xlabel('Y capture position / mm',fontsize=20)
        ax2.set_ylabel('Frequency',fontsize=20)

        ax3=fig.add_subplot(gs[2,:])
        zhist=ax3.hist(indetz,bins=250)
        ax3.set_xlabel('Z capture position / mm',fontsize=20)
        ax3.set_ylabel('Frequency',fontsize=20)

        fig.suptitle(self.source,fontsize=20)
        if plotdir!=None:
            fig.savefig(plotdir+"/"+self.source+"_abs_cap_pos_dist.png")
        plt.show()

    def ScatterTThresh(self,plotdir=None,log=True,standard=False,upperlim=20):
        """
        Plots the 1D histogram of time between scatter, for scatters above the detectable energy threshold.

        PARAMETERS
        -----------
        plotdir : str
            Directory to save the plot to. If None, the plot is not saved.

        log : bool
            If True, plot the logarithmic histogram

        standard : bool
            If True, for a non-logarithmic histogram, set the upper limit to the 
            median + 3 times the median absolute deviation (defined in Scipy)

        upperlim : int
            For a non-logarithmic histogram and standard = False, sets the upper limit.
        """

        e=self.track['EnergyChange'].explode()*-1
        t=self.track['StepTime'].explode()

        data=pd.concat((t,e),axis=1)
        data['TimeDiff']=data['StepTime'].diff()
        locs=[]
        inds=np.unique(data.index.get_level_values(0))
        for ind in inds:
            locs.append(np.where(data.index==ind)[0][0])
        data['TimeDiff'].iloc[locs]=data['StepTime'].iloc[locs]
        data['Detectable']=data['EnergyChange']>4.8e5
        tplot=data['TimeDiff'][data['Detectable']]
        med=np.median(tplot)
        std=median_absolute_deviation(tplot)

        if log:
            bins=10**np.linspace(np.log10(min(tplot)),np.log10(max(tplot)),200)
        else:
            bins=200

        if log:
            rang=(min(tplot),max(tplot))
        elif standard:
            rang=(0,med+5*std)
        else:
            rang=(0,upperlim)
        fig=plt.figure(figsize=(19.2,10.8))
        ax=fig.add_subplot(111)
        ax.hist(tplot,bins=bins,ec='b',range=rang)
        if log:
            ax.set_xscale('log')
        ax.set_xlabel('Time between scatter / ns',fontsize=20)
        ax.set_ylabel('Frequency',fontsize=20)
        if log:
            append=", log scale"
        else:
            append=""
        ax.set_title(self.source + " time between scatters above threshold"+append,fontsize=24)
        if plotdir:
            fig.savefig(plotdir+"/"+self.source+"scatter_time_thresh_dist.png")
        plt.show()
        return

    def DistancePlots(self,first=False):
        xpos=self.track['XPosition'].explode()
        ypos=self.track['YPosition'].explode()
        zpos=self.track['ZPosition'].explode()
        indet=(xpos>self.xmin)&(xpos<self.xmax)&(ypos>self.ymin)&(ypos<self.ymax)&(zpos>self.zmin)&(zpos<self.zmax)
        xpos=xpos[indet]
        ypos=ypos[indet]
        zpos=zpos[indet]

        capx=self.final['CapPosX']
        capy=self.final['CapPosY']
        capz=self.final['CapPosZ']
        data=pd.concat((self.track['EnergyChange'],capx,capy,capz),axis=1)

        data['capindet']=(capx>self.xmin)&(capy<self.xmax)&(capy>self.ymin)&(capy<self.ymax)&(capz>self.zmin)&(capz<self.zmax)
        data=data.explode('EnergyChange')[indet]

        dist=pd.concat((data,xpos,ypos,zpos),axis=1)
        dist=dist[dist['capindet']]
        dist=dist[dist['EnergyChange']<=-4.8e5]

        if first:
            ind=0
        else:
            ind=-1

        inds=[np.where(dist.index==s)[0][ind] for s in np.unique(dist.index)]
        dist=dist.iloc[inds]

        dist['ScattToCap']=((dist['XPosition']-dist['CapPosX'])**2 + (dist['YPosition']-dist['CapPosY'])**2 + (dist['ZPosition']-dist['CapPosZ'])**2)**(1/2)
        dbins=np.linspace(0,500,100)

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.hist(dist['ScattToCap'],bins=dbins)
        if first:
            ax.set_xlabel("Distance from first detectable scatter to capture",fontsize=20)
        else:
            ax.set_xlabel("Distance from last detectable scatter to capture",fontsize=20)
        ax.set_ylabel('Frequency',fontsize=20)
        ax.set_title(self.source + " distance from scatter to capture",fontsize=24)
        plt.show()

    def ScattToCap(self,num):
        if self.cubedist is not None:
            dists=self.cubedist
        else:
            raise TypeError("Cube distribution not calculated. Please run the 'CubeDistPrep' function first.")
            

        caps=dists.loc[:,['XCapCube','YCapCube','ZCapCube']].values
        scatts=dists.loc[:,['XScattCube','YScattCube','ZScattCube']].values

        capsarr=np.zeros([4,4,4])
        scattarr=np.zeros([4,4,4])

        for i in range(len(caps)):
            capsarr[caps[i][0],caps[i][1],caps[i][2]]+=1
            scattarr[scatts[i][0],scatts[i][1],scatts[i][2]]+=1
            
        datarr=np.concatenate([capsarr[:,:,:,np.newaxis],scattarr[:,:,:,np.newaxis]],axis=3)
        data=np.concatenate([caps[:,:,np.newaxis],scatts[:,:,np.newaxis]],axis=2)
        data=data+0.5+(0.888/50)*data
        plotarr=np.zeros([4,4,4])

        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        sc=0.888/50
        subject=np.ones((4,4,4))
        plotter=ndap.NDArrayPlotter(subject,spacing=("even",sc,sc,sc))
        plotter.set_alpha(0.05)
        cmap=cm.viridis
        norm=colors.Normalize(vmin=np.min(plotarr),vmax=np.max(plotarr))
        plotter.colors=cmap(norm(plotarr))
        plotter.render(azim=-45,elev=25,ax=ax)

        for i in range(num):
            ax.quiver(data[i,0,0],data[i,1,0],data[i,2,0],
                      data[i,0,1]-data[i,0,0],data[i,1,1]-data[i,1,0],data[i,2,1]-data[i,2,0],
                      lw=2,arrow_length_ratio=0.2)

        plt.show()


    def CubeDistPrep(self, rot=True):
        capindet=((self.final['CapPosX']>self.xmin)&(self.final['CapPosX']<self.xmax)&(self.final['CapPosY']>self.ymin)&(self.final['CapPosY']<self.ymax)&(self.final['CapPosZ']>self.zmin)&(self.final['CapPosZ']<self.zmax)).rename('capindet')
        xpos=self.track['XPosition'].explode()
        ypos=self.track['YPosition'].explode()
        zpos=self.track['ZPosition'].explode()

        indet=(xpos>self.xmin)&(xpos<self.xmax)&(ypos>self.ymin)&(ypos<self.ymax)&(zpos>self.zmin)&(zpos<self.zmax)
        xpos,ypos,zpos=xpos[indet],ypos[indet],zpos[indet]

        cdf=pd.concat((self.detresponse['cubex'].explode(),self.detresponse['cubey'].explode(),self.detresponse['cubez'].explode()),axis=1)
        cinds=[np.where(cdf.index==s)[0][-1] for s in np.unique(cdf.index)]
        cdf=cdf.iloc[cinds]

        dist=pd.concat((self.track['StepTime'],self.final.loc[:,['CaptureTime','CapPosX','CapPosY','CapPosZ','CapCube']],capindet,cdf,self.initial['energy'].explode()),axis=1).explode('StepTime')[indet]
        dist=pd.concat((dist,xpos,ypos,zpos,self.track['EnergyChange'].explode()[indet]),axis=1)
        dist=dist[dist['capindet']]
        dist=dist[dist['EnergyChange']<=-4.8e5]

        dist=pd.concat((dist,((dist['CapPosX']-self.detloc[0]+2*self.cubesize)/self.cubesize).astype(int).rename("XCapCube")),axis=1)
        dist=pd.concat((dist,((dist['CapPosY']-self.detloc[1]+2*self.cubesize)/self.cubesize).astype(int).rename("YCapCube")),axis=1)
        dist=pd.concat((dist,((dist['CapPosZ']-self.detloc[2]+2*self.cubesize)/self.cubesize).astype(int).rename("ZCapCube")),axis=1)

        inds=[np.where(dist.index==s)[0][0] for s in np.unique(dist.index)]
        dists=dist.iloc[inds]

        dists=pd.concat((dists,((dists['XPosition']-self.detloc[0]+2*self.cubesize)/self.cubesize).astype(int).rename("XScattCube")),axis=1)
        dists=pd.concat((dists,((dists['YPosition']-self.detloc[1]+2*self.cubesize)/self.cubesize).astype(int).rename("YScattCube")),axis=1)
        dists=pd.concat((dists,((dists['ZPosition']-self.detloc[2]+2*self.cubesize)/self.cubesize).astype(int).rename("ZScattCube")),axis=1)

        dists['cubex']-=1
        dists['cubey']-=1
        dists['cubez']-=1

        if rot:
            dists['XScattCube']=dists['XScattCube']*-1+3
            dists['YScattCube']=dists['YScattCube']*-1+3

            dists['DeltaX']=dists['cubex']-dists['XScattCube']
            dists['DeltaY']=dists['cubey']-dists['YScattCube']
            dists['DeltaZ']=dists['cubez']-dists['ZScattCube']

        else:
            dists['DeltaX']=(dists['XCapCube']-dists['XScattCube'])
            dists['DeltaY']=(dists['YCapCube']-dists['YScattCube'])
            dists['DeltaZ']=(dists['ZCapCube']-dists['ZScattCube'])


        self.cubedist=dists
        return

    def DeltaPlots(self):
        if self.cubedist is not None:
            dists=self.cubedist
        else:
            raise TypeError("Cube distribution not calculated. Please run the 'CubeDistPrep' function first.")

        #xbins=(np.linspace(min(dists['DeltaX']),max(dists['DeltaX']),max(dists['DeltaX'])-min(dists['DeltaX'])+1))-0.5
        #ybins=(np.linspace(min(dists['DeltaY']),max(dists['DeltaY']),max(dists['DeltaY'])-min(dists['DeltaY'])+1))-0.5
        #zbins=(np.linspace(min(dists['DeltaZ']),max(dists['DeltaZ']),max(dists['DeltaZ'])-min(dists['DeltaZ'])+1))-0.5
        xbins=np.linspace(-3,4,8)-0.5
        ybins=np.linspace(-3,4,8)-0.5
        zbins=np.linspace(-3,4,8)-0.5


        fig,ax=plt.subplots(1,3,figsize=(30,10))

        n1,bins1,patches1=ax[0].hist(dists['DeltaX'],bins=xbins,ec='b')
        ax[0].set_title("Delta X",fontsize=24)
        ax[0].set_xlabel('X Cubes from Scatter to Capture',fontsize=20)
        ax[0].set_ylabel('Frequency',fontsize=20)

        n2,bins2,patches2=ax[1].hist(dists['DeltaY'],bins=ybins,ec='b')
        ax[1].set_title("Delta Y",fontsize=24)
        ax[1].set_xlabel('Y Cubes from Scatter to Capture',fontsize=20)
        ax[1].set_ylabel('Frequency',fontsize=20)

        n3,bins3,patches3=ax[2].hist(dists['DeltaZ'],bins=zbins,ec='b')
        ax[2].set_title("Delta Z",fontsize=24)
        ax[2].set_xlabel('Z Cubes from Scatter to Capture',fontsize=20)
        ax[2].set_ylabel('Frequency',fontsize=20)

        dd=np.hstack((n1,n2,n3)).flatten()
        yl=1.1*max(dd)
        ax[0].set_ylim(top=yl)
        ax[1].set_ylim(top=yl)
        ax[2].set_ylim(top=yl)

        for axs in ax:
            axs.xaxis.set_major_locator(MultipleLocator(1))
            axs.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        fig.suptitle(self.source,fontsize=24)

        plt.show()

    def CubesAnalysis(self,detresponse=False,rot=False,normalise=False,rot_axis=2):
        cubes=self.detresponse.loc[:,['cubex','cubey','cubez']]
        capindet=((self.final['CapPosX']>self.xmin)&(self.final['CapPosX']<self.xmax)&(self.final['CapPosY']>self.ymin)&(self.final['CapPosY']<self.ymax)&(self.final['CapPosZ']>self.zmin)&(self.final['CapPosZ']<self.zmax)).rename('capindet')
        cdf=pd.concat((cubes['cubex'].explode(),cubes['cubey'].explode(),cubes['cubez'].explode()),axis=1)-1
        cinds=[np.where(cdf.index==s)[0][-1] for s in np.unique(cdf.index)]
        finds=[np.where(cdf.index==s)[0][0] for s in np.unique(cdf.index)]
        fdf=cdf.iloc[finds]
        fdf.columns=['scattcubex','scattcubey','scattcubez']
        cdf=cdf.iloc[cinds]
        
        xpos=self.track['XPosition'].explode()
        ypos=self.track['YPosition'].explode()
        zpos=self.track['ZPosition'].explode()

        indet=(xpos>self.xmin)&(xpos<self.xmax)&(ypos>self.ymin)&(ypos<self.ymax)&(zpos>self.zmin)&(zpos<self.zmax)
        xpos,ypos,zpos=xpos[indet],ypos[indet],zpos[indet]
        dist=pd.concat((self.track['StepTime'],self.final.loc[:,['CaptureTime','CapPosX','CapPosY','CapPosZ','CapCube']],capindet,cdf,fdf,self.initial['energy'].explode()),axis=1).explode('StepTime')[indet]
        dist=pd.concat((dist,xpos,ypos,zpos,self.track['EnergyChange'].explode()[indet]),axis=1)
        dist=dist[dist['capindet']]
        dist=dist[dist['EnergyChange']<=-4.8e5]
        
        dist=pd.concat((dist,((dist['CapPosX']-self.detloc[0]+2*self.cubesize)/self.cubesize).astype(int).rename("XCapCube")),axis=1)
        dist=pd.concat((dist,((dist['CapPosY']-self.detloc[1]+2*self.cubesize)/self.cubesize).astype(int).rename("YCapCube")),axis=1)
        dist=pd.concat((dist,((dist['CapPosZ']-self.detloc[2]+2*self.cubesize)/self.cubesize).astype(int).rename("ZCapCube")),axis=1)
        
        inds=[np.where(dist.index==s)[0][0] for s in np.unique(dist.index)]
        dists=dist.iloc[inds]
        dists=pd.concat((dists,((dists['XPosition']-self.detloc[0]+2*self.cubesize)/self.cubesize).astype(int).rename("XScattCube")),axis=1)
        dists=pd.concat((dists,((dists['YPosition']-self.detloc[1]+2*self.cubesize)/self.cubesize).astype(int).rename("YScattCube")),axis=1)
        dists=pd.concat((dists,((dists['ZPosition']-self.detloc[2]+2*self.cubesize)/self.cubesize).astype(int).rename("ZScattCube")),axis=1)
        if rot:
            if rot_axis==0:
                dists['YScattCube']=dists['YScattCube']*-1+3
                dists['ZScattCube']=dists['ZScattCube']*-1+3
            if rot_axis==1:
                dists['XScattCube']=dists['XScattCube']*-1+3
                dists['ZScattCube']=dists['ZScattCube']*-1+3
            if rot_axis==2:
                dists['XScattCube']=dists['XScattCube']*-1+3
                dists['YScattCube']=dists['YScattCube']*-1+3
        if detresponse:
            dists['DeltaX']=(dists['cubex']-dists['scattcubex'])
            dists['DeltaY']=(dists['cubey']-dists['scattcubey'])
            dists['DeltaZ']=(dists['cubez']-dists['scattcubez'])
        else:
            dists['DeltaX']=(dists['cubex']-dists['XScattCube'])
            dists['DeltaY']=(dists['cubey']-dists['YScattCube'])
            dists['DeltaZ']=(dists['cubez']-dists['ZScattCube'])
        if normalise:
            norms=np.linalg.norm(dists.loc[:,['DeltaX','DeltaY','DeltaZ']].values.astype(int),axis=1)
            dists['DeltaX']/=norms
            dists['DeltaY']/=norms
            dists['DeltaZ']/=norms
        if rot:
            if rot_axis==0:
                dists['DeltaY']*=-1
                dists['DeltaZ']*=-1
            if rot_axis==1:
                dists['DeltaX']*=-1
                dists['DeltaZ']*=-1
            if rot_axis==2:
                dists['DeltaX']*=-1
                dists['DeltaY']*=-1

        self.cubedist=dists

    
class SimComparison():
    def __init__(self):
        self.dists=[]
    
    def add_dist(self,name,source,detloc=np.array([0,-1627,1500]),cubesize=50.888,rot=False,rot_axis=2):
        self.dists.append(SimEventViewer(name,source,detloc,cubesize))
        self.dists[-1].CubesAnalysis(rot=rot,rot_axis=rot_axis)

    def calc_accuracy(self,truevec,sample_size=50):
        vecs=[]
        for dist in self.dists:
            vecs.append(dist.cubedist.loc[:,['DeltaX','DeltaY','DeltaZ']].values.astype(int))
        for i in range(len(vecs)):
            vecs[i]=vecs[i]/np.linalg.norm(vecs[i],axis=1)[:,np.newaxis]
        
        inds=[]
        samples=[]
        for i in range(len(vecs)):
            inds.append(np.arange(0,len(vecs[i]),1))
            samples.append(np.random.choice(inds[i],sample_size,replace=False))
            inds[i]=np.delete(inds[i],samples[i])
            if i==0:
                data=vecs[i][samples[i]]
            else:
                data=np.concatenate((data,vecs[i][samples[i]]),axis=0)
        
        meanvec=np.nanmean(data,axis=0)
        meanvec=meanvec/np.linalg.norm(meanvec)
        deviations=np.hstack((meanvec-truevec,len(data)))
        while min([len(ind) for ind in inds])>sample_size:
            samples=[]
            for i in range(len(vecs)):
                samples.append(np.random.choice(inds[i],sample_size,replace=False))
                inds[i]=np.delete(inds[i],np.searchsorted(inds[i],samples[i]))
                data=np.concatenate((data,vecs[i][samples[i]]),axis=0)

            meanvec=np.nanmean(data,axis=0)
            meanvec=meanvec/np.linalg.norm(meanvec)
            comp=np.hstack((meanvec-truevec,len(data)))
            deviations=np.vstack((deviations,comp))

        return deviations
    



class MonoenergeticDistViewer():
    def __init__(self,folder,detloc=np.array([0.,-1627.,1500.]),cubesize=50.888,distance=1.5,loadfile=True):
        self.detloc=detloc
        self.distance=distance
        self.cubesize=cubesize
        self.xmin,self.xmax=detloc[0]-2*cubesize,detloc[0]+2*cubesize
        self.ymin,self.ymax=detloc[1]-2*cubesize,detloc[1]+2*cubesize
        self.zmin,self.zmax=detloc[2]-2*cubesize,detloc[2]+2*cubesize

        if loadfile:
            self.df=pd.read_hdf(folder+"/monos.h5")
            self.es=np.unique(self.df['SourceEnergy'])
            return

        files=[name for name in os.listdir(folder) if name[-6:]=="kev.h5" or name[-6:]=="MeV.h5" or name[-6:]=="keV.h5"]
        arrs=['cubex','cubey','cubez','StepTime','StepLength','EnergyChange','CapPosX','CapPosY','CapPosZ','CaptureTime','XPosition','YPosition','ZPosition','energy']
        energies={'kev':1000,'keV':1000,'MeV':1e6}
        name=files.pop()
        df=pd.read_hdf(folder+"/"+name).loc[:,arrs]
        ind=df.index
        e=int(name[5:-6])*energies[name[-6:-3]]
        #mind=[[e]*len(ind),ind]
        #t=list(zip(*mind))
        #index=pd.MultiIndex.from_tuples(t,names=['Energy / eV','index'])
        #df=file0.set_index(index)
        df['SourceEnergy']=[e]*len(ind)
        for file in files:
            data=pd.read_hdf(folder+"/"+file).loc[:,arrs]
            ind=data.index
            energy=int(file[5:-6])*energies[file[-6:-3]]
            #mind=[[energy]*len(ind),ind]
            #t=list(zip(*mind))
            #index=pd.MultiIndex.from_tuples(t,names=['Energy / eV','index'])
            #data=data.set_index(index)
            data['SourceEnergy']=[energy]*len(ind)
            df=pd.concat((df,data))
        df.reset_index(inplace=True,drop=True)
        cx=((df['CapPosX']>self.xmin) & (df['CapPosX']<self.xmax))
        cy=((df['CapPosY']>self.ymin) & (df['CapPosY']<self.ymax))
        cz=((df['CapPosZ']>self.zmin) & (df['CapPosZ']<self.zmax))
        df['capindet']=cx & cy & cz

        nscatt=[]
        nscattthresh=[]
        for i in range(len(df)):
            locdf=df.iloc[i]
            xin=np.where((locdf['XPosition'] > self.xmin) & (locdf['XPosition'] < self.xmax))[0]
            yin=np.where((locdf['YPosition'] > self.ymin) & (locdf['YPosition'] < self.ymax))[0]
            zin=np.where((locdf['ZPosition'] > self.zmin) & (locdf['ZPosition'] < self.zmax))[0]
            detect=np.where(locdf['EnergyChange']<=-480e3)[0]
            inn=np.intersect1d(np.intersect1d(xin,yin),zin)
            comb=np.intersect1d(inn,detect)
            nscatt.append(len(inn))
            nscattthresh.append(len(comb))
        df['NScatt']=nscatt
        df['NScattOverThresh']=nscattthresh
        diffs=[]
        for t in df['StepTime']:
            t_roll=np.roll(np.append(np.diff(t),t[0]),1)
            diffs.append(t_roll)
        df['TSteps']=diffs
        self.df = df

        self.es=np.unique(df['SourceEnergy'])
        if not os.path.isfile(folder+'/monos.h5'):
            df.to_hdf(folder+'/monos.h5',key='monos')
        return


    def CapTimeDist(self,log=False):
        data=self.df.loc[:,['CaptureTime','SourceEnergy']]
        self.curr_pos=0
        fig=plt.figure()
        ax=fig.add_subplot(111)
        es=self.es
        self.curr_pos=0
        def key_event(e):
            if e.key=="right":
                self.curr_pos+=1
                self.curr_pos%=len(es)
                capt=data['CaptureTime'][data['SourceEnergy']==es[self.curr_pos]]
                ax.cla()
                if log:
                    bins=10**np.linspace(np.log10(min(capt)),np.log10(max(capt)),200)
                else:
                    bins=100
                ax.hist(capt,bins=bins,range=(0,1e6))
                if log:
                    ax.set_xscale('log')
                ax.set_ylim(0,400)
                ax.set_xlabel('Capture Time / ns',fontsize=20)
                ax.set_ylabel('Frequency',fontsize=20)

                if es[self.curr_pos]>=1e6:
                    title="Monoenergetic " + str(int(es[self.curr_pos]/1e6))+" MeV"
                elif es[self.curr_pos]>=1e3 and es[self.curr_pos]<1e6:
                    title="Monoenergetic " + str(int(es[self.curr_pos]/1e3))+" keV"
                ax.set_title(title,fontsize=20)
                fig.canvas.draw()
            elif e.key=="left":
                self.curr_pos-=1
                self.curr_pos%=len(es)
                capt=data['CaptureTime'][data['SourceEnergy']==es[self.curr_pos]]
                ax.cla()
                if log:
                    bins=10**np.linspace(np.log10(min(capt)),np.log10(max(capt)),200)
                else:
                    bins=100
                ax.hist(capt,bins=bins,range=(0,1e6))
                ax.set_ylim(0,400)
                ax.set_xlabel('Capture Time / ns',fontsize=20)
                ax.set_ylabel('Frequency',fontsize=20)

                if es[self.curr_pos]>=1e6:
                    title="Monoenergetic " + str(int(es[self.curr_pos]/1e6))+" MeV"
                elif es[self.curr_pos]>=1e3 and es[self.curr_pos]<1e6:
                    title="Monoenergetic " + str(int(es[self.curr_pos]/1e3))+" keV"
                ax.set_title(title,fontsize=20)
                fig.canvas.draw()
            else:
                return
    
        fig.canvas.mpl_connect('key_press_event',key_event)
        capt=data['CaptureTime'][data['SourceEnergy']==es[self.curr_pos]]
        if log:
            bins=10**np.linspace(np.log10(min(capt)),np.log10(max(capt)),200)
        else:
            bins=100
        ax.hist(capt,bins=bins,range=(0,1e6))
        ax.set_xlabel('Capture Time / ns',fontsize=20)
        if log:
            ax.set_xscale('log')
        ax.set_ylabel('Frequency',fontsize=20)
        ax.set_ylim(0,400)
        if es[self.curr_pos]>=1e6:
            title="Monoenergetic " + str(int(es[self.curr_pos]/1e6))+" MeV"
        elif es[self.curr_pos]>=1e3 and es[self.curr_pos]<1e6:
            title="Monoenergetic " + str(int(es[self.curr_pos]/1e3))+" keV"
        ax.set_title(title,fontsize=20)
        plt.show()

    def CapPosDist(self):
        capx=self.df.loc[:,['CapPosX','SourceEnergy']]
        capy=self.df.loc[:,['CapPosY','SourceEnergy']]
        capz=self.df.loc[:,['CapPosZ','SourceEnergy']]
        es=self.es

        fig=plt.figure()
        gs=fig.add_gridspec(3,5)
        ax1=fig.add_subplot(gs[0,:])
        ax2=fig.add_subplot(gs[1,:])
        ax3=fig.add_subplot(gs[2,:])
        fig.tight_layout()
        self.curr_pos=0
        def key_event(e):
            if e.key=="right":
                self.curr_pos+=1
                self.curr_pos%=len(self.es)
                ax1.cla()
                ax2.cla()
                ax3.cla()
                
                ax1.hist(capx['CapPosX'][capx['SourceEnergy']==es[self.curr_pos]],bins=250,range=(self.xmin-5,self.xmax+5))
                ax2.hist(capy['CapPosY'][capy['SourceEnergy']==es[self.curr_pos]],bins=250,range=(self.ymin-5,self.ymax+5))
                ax3.hist(capz['CapPosZ'][capz['SourceEnergy']==es[self.curr_pos]],bins=250,range=(self.zmin-5,self.zmax+5))
                ax1.set_xlabel('X capture position / mm',fontsize=20)
                ax1.set_ylabel('Frequency',fontsize=20)
                ax2.set_xlabel('Y capture position / mm',fontsize=20)
                ax2.set_ylabel('Frequency',fontsize=20)
                ax3.set_xlabel('Z capture position / mm',fontsize=20)
                ax3.set_ylabel('Frequency',fontsize=20)
                ax1.set_ylim(0,200)
                ax2.set_ylim(0,200)
                ax3.set_ylim(0,200)
                if es[self.curr_pos] >= 1e6:
                    fig.suptitle("Monoenergetic " + str(int(es[self.curr_pos]/1e6)) + " MeV",y=0.95,fontsize=24)
                elif es[self.curr_pos] >= 1e3 and es[self.curr_pos] < 1e6:
                    fig.suptitle("Monoenergetic " + str(int(es[self.curr_pos]/1e3)) + " keV",y=0.95,fontsize=24)
                fig.tight_layout()
                fig.canvas.draw()
            elif e.key=="left":
                self.curr_pos-=1
                self.curr_pos%=len(self.es)
                ax1.cla()
                ax2.cla()
                ax3.cla()
                
                ax1.hist(capx['CapPosX'][capx['SourceEnergy']==es[self.curr_pos]],bins=250,range=(self.xmin-5,self.xmax+5))
                ax2.hist(capy['CapPosY'][capy['SourceEnergy']==es[self.curr_pos]],bins=250,range=(self.ymin-5,self.ymax+5))
                ax3.hist(capz['CapPosZ'][capz['SourceEnergy']==es[self.curr_pos]],bins=250,range=(self.zmin-5,self.zmax+5))
                ax1.set_xlabel('X capture position / mm',fontsize=20)
                ax1.set_ylabel('Frequency',fontsize=20)
                ax2.set_xlabel('Y capture position / mm',fontsize=20)
                ax2.set_ylabel('Frequency',fontsize=20)
                ax3.set_xlabel('Z capture position / mm',fontsize=20)
                ax3.set_ylabel('Frequency',fontsize=20)
                ax1.set_ylim(0,200)
                ax2.set_ylim(0,200)
                ax3.set_ylim(0,200)
                if es[self.curr_pos] >= 1e6:
                    fig.suptitle("Monoenergetic " + str(int(es[self.curr_pos]/1e6)) + " MeV",y=0.95,fontsize=24)
                elif es[self.curr_pos] >= 1e3 and es[self.curr_pos] < 1e6:
                    fig.suptitle("Monoenergetic " + str(int(es[self.curr_pos]/1e3)) + " keV",y=0.95,fontsize=24)
                fig.tight_layout()
                fig.canvas.draw()
            else:
                return
        
        fig.canvas.mpl_connect("key_press_event",key_event)

        ax1.hist(capx['CapPosX'][capx['SourceEnergy']==es[self.curr_pos]],bins=250,range=(self.xmin-5,self.xmax+5))
        ax2.hist(capy['CapPosY'][capy['SourceEnergy']==es[self.curr_pos]],bins=250,range=(self.ymin-5,self.ymax+5))
        ax3.hist(capz['CapPosZ'][capz['SourceEnergy']==es[self.curr_pos]],bins=250,range=(self.zmin-5,self.zmax+5))
        ax1.set_xlabel('X capture position / mm',fontsize=20)
        ax1.set_ylabel('Frequency',fontsize=20)
        ax2.set_xlabel('Y capture position / mm',fontsize=20)
        ax2.set_ylabel('Frequency',fontsize=20)
        ax3.set_xlabel('Z capture position / mm',fontsize=20)
        ax3.set_ylabel('Frequency',fontsize=20)
        ax1.set_ylim(0,200)
        ax2.set_ylim(0,200)
        ax3.set_ylim(0,200)
        if es[0]>=1e6:
            fig.suptitle("Monoenergetic " + str(int(es[0]/1e6)) + " MeV",y=0.95,fontsize=24)
        elif es[0]>=1e3 and es[0] < 1e6:
            fig.suptitle("Monoenergetic " + str(int(es[0]/1e3)) + " keV",y=0.95,fontsize=24)
        plt.show()

    def NScatt(self,threshold=False,inc_capture=True,log=False):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        if threshold:
            thresh='NScattOverThresh'
            bins=5
            rang=(0,5)
        else:
            thresh='NScatt'
            if log:
                bins=150
                rang=(0,150)
            else:
                bins=50
                rang=(0,80)
        counts=np.unique(self.df['SourceEnergy'],return_counts=True)[1]
        self.curr_pos=0
        def key_event(e):
            if e.key=="right":
                self.curr_pos+=1
                self.curr_pos%=len(self.es)
                ax.cla()
                ax.hist(self.df[thresh][self.df['SourceEnergy']==self.es[self.curr_pos]],bins=bins,color='purple',range=rang,alpha=0.3,label='All neutrons')
                if inc_capture:
                    ax.hist(self.df[thresh][self.df['capindet']][self.df['SourceEnergy']==self.es[self.curr_pos]],bins=bins,color='purple',range=rang,alpha=0.7,label='Captured neutrons')
                if log:
                    ax.set_yscale('log')
                
                if threshold:
                    ax.set_xlabel('Number of scatters in detector over detection threshold (480 keV)',fontsize=20)
                    ax.set_xlim(0,5)
                else:
                    ax.set_xlabel('Number of scatters in detector',fontsize=20)
                    ax.set_ylim(top=0.6*max(counts))
                ax.set_ylabel('Frequency',fontsize=20)

                if self.es[self.curr_pos]>=1e6:
                    ax.set_title('Monoenergetic ' + str(int(self.es[self.curr_pos]/1e6)) + " MeV",fontsize=24)
                elif self.es[self.curr_pos]>=1e3 and self.es[self.curr_pos]<1e6:
                    ax.set_title('Monoenergetic ' + str(int(self.es[self.curr_pos]/1e3)) + " keV",fontsize=24)
                elif self.es[self.curr_pos]<1e3:
                    ax.set_title("Monoenergetic " + str(int(self.es[self.curr_pos])) + " eV",fontsize=24)
                if inc_capture:
                    ax.legend(loc='upper right')
                fig.canvas.draw()
            elif e.key=="left":
                self.curr_pos-=1
                self.curr_pos%=len(self.es)
                ax.cla()
                ax.hist(self.df[thresh][self.df['SourceEnergy']==self.es[self.curr_pos]],bins=bins,color='purple',alpha=0.3,range=rang,label='All neutrons')
                if inc_capture:
                    ax.hist(self.df[thresh][self.df['capindet']][self.df['SourceEnergy']==self.es[self.curr_pos]],bins=bins,color='purple',alpha=0.7,range=rang,label='Captured neutrons')
                if log:
                    ax.set_yscale('log')                
                if threshold:
                    ax.set_xlabel('Number of scatters in detector over detection threshold (480 keV)',fontsize=20)
                    ax.set_xlim(0,5)
                else:
                    ax.set_xlabel('Number of scatters in detector',fontsize=20)
                    ax.set_ylim(top=0.6*max(counts))

                ax.set_ylabel('Frequency',fontsize=20)
                if self.es[self.curr_pos]>=1e6:
                    ax.set_title('Monoenergetic ' + str(int(self.es[self.curr_pos]/1e6)) + " MeV",fontsize=24)
                elif self.es[self.curr_pos]>=1e3 and self.es[self.curr_pos]<1e6:
                    ax.set_title('Monoenergetic ' + str(int(self.es[self.curr_pos]/1e3)) + " keV",fontsize=24)
                elif self.es[self.curr_pos]<1e3:
                    ax.set_title("Monoenergetic " + str(int(self.es[self.curr_pos])) + " eV",fontsize=24)
                if inc_capture:
                    ax.legend(loc='upper right')
                fig.canvas.draw()
            else:
                return
            
        fig.canvas.mpl_connect("key_press_event",key_event)
        ax.hist(self.df[thresh][self.df['SourceEnergy']==self.es[self.curr_pos]],bins=bins,color='purple',alpha=0.3,range=rang,label='All neutrons')
        if inc_capture:
            ax.hist(self.df[thresh][self.df['capindet']][self.df['SourceEnergy']==self.es[self.curr_pos]],bins=bins,color='purple',alpha=0.7,range=rang,label='Captured neutrons')
        if log:
            ax.set_yscale('log')        
        if threshold:
            ax.set_xlabel('Number of scatters in detector over detection threshold (480 keV)',fontsize=20)
            ax.set_xlim(0,5)
        else:
            ax.set_xlabel('Number of scatters in detector',fontsize=20)
            ax.set_ylim(top=0.6*max(counts))
            ax.set_xlim(0,80)

        ax.set_ylabel('Frequency',fontsize=20)
        if self.es[0]>=1e6:
            ax.set_title('Monoenergetic ' + str(int(self.es[0]/1e6)) + " MeV",fontsize=24)
        elif self.es[0]>=1e3 and self.es[0]<1e6:
            ax.set_title('Monoenergetic ' + str(int(self.es[0]/1e3)) + " keV",fontsize=24)
        elif self.es[0]<1e3:
            ax.set_title("Monoenergetic " + str(int(self.es[0])) + " eV",fontsize=24)
        if inc_capture:
            ax.legend(loc='upper right')
        plt.show()

    def ScatterTEloss(self,cut_therm=True,strict_cut=True):
        xpos=self.df['XPosition'].explode()
        ypos=self.df['YPosition'].explode()
        zpos=self.df['ZPosition'].explode()
        indet=(xpos>self.xmin) & (xpos<self.xmax) & (ypos>self.ymin) & (ypos<self.ymax) & (zpos>self.zmin) & (zpos<self.zmax)
        xpos,ypos,zpos=None,None,None

        if cut_therm:
            if strict_cut:
                thresh=1
            else:
                thresh=0.005
        else:
            thresh=0

        dist=pd.concat((self.df.loc[:,['TSteps','capindet','SourceEnergy']].explode('TSteps')[indet],self.df['EnergyChange'].explode()[indet]*-1),axis=1)
        eloss=dist['EnergyChange']>thresh
        dist=dist[eloss]

        fig=plt.figure(figsize=(19.2,10.8))
        gs=fig.add_gridspec(2,4)
        ax1=fig.add_subplot(gs[:,:2])
        ax2=fig.add_subplot(gs[0,2:])
        ax3=fig.add_subplot(gs[1,2:])

        self.curr_pos=0
        us=np.full(200,1000)
        ns=np.full(200,1)
        thermal=np.full(200,0.025)
        detect=np.full(200,480000)
        

        def key_event(e):
            if e.key=="right":
                self.curr_pos+=1
                self.curr_pos%=len(self.es)
                ax1.cla()
                ax2.cla()
                ax3.cla()

                data=dist[dist['SourceEnergy']==self.es[self.curr_pos]]
                tbins=10**np.linspace(-5,5,200)
                ebins=10**np.linspace(np.log10(min(data['EnergyChange'])),np.log10(max(data['EnergyChange'])),200)

                hist=ax1.hist2d(data['TSteps'],data['EnergyChange'],bins=[tbins,ebins],norm=colors.LogNorm())
                ax1.set_xlabel('Time between scatter / ns',fontsize=20)
                ax1.set_ylabel('Energy loss in scatter',fontsize=20)
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                ax1.set_ylim(top=10e7)
                labels=ax1.get_yticks().tolist()
                for i in range(len(labels)):
                    if labels[i]<1000:
                        labels[i]=str(labels[i]) + " eV"
                    elif labels[i]<1e6:
                        labels[i]=str(labels[i]/1e3) + " keV"
                    elif labels[i] >= 1e6:
                        labels[i]=str(labels[i]/1e6) + " MeV"
                ax1.set_yticklabels(labels)
                ax1.set_xlim(1e-5,1e5)
                ax1.fill_between(tbins,0,detect,alpha=0.3,label='Not detectable ($\\textless$ 480 keV)')
                ax1.plot(us,ebins,label='1 $\mu$s')
                ax1.plot(ns,ebins,label='1 ns')
                ax1.plot(tbins,thermal,label='Thermal neutron energy (0.025 eV)')
                ax1.legend(loc='lower left')
                cbar.set_clim(vmin=0,vmax=np.max(hist[0]))
                cbar.draw_all()
                dbins=np.linspace(0,500,100)
                visible=data['EnergyChange']>4.8e5

                ax2.hist(data['TSteps'],bins=tbins,color='darkblue',alpha=0.3,label='All events')
                ax2.hist(data['TSteps'][visible],bins=tbins,color='purple',alpha=0.8,label='Detectable events')
                ax2.hist(data['TSteps'][data['capindet']],bins=tbins,color='purple',alpha=0.5,label='Captured')
                ax2.set_xscale('log')
                ax2.set_xlabel('Time between scatter / ns',fontsize=20)
                ax2.set_ylabel('Frequency',fontsize=20)
                ax2.legend(loc='lower left')

                ax3.hist(data['EnergyChange'],bins=ebins,color='darkblue',alpha=0.3,label='All events')
                ax3.hist(data['EnergyChange'][visible],bins=ebins,color='purple',alpha=0.8,label='Detectable events')
                ax3.hist(data['EnergyChange'][data['capindet']],bins=ebins,color='purple',alpha=0.5,label='Captured')
                ax3.set_xscale('log')
                labels=ax3.get_xticks().tolist()
                for i in range(len(labels)):
                    if labels[i]<1000:
                        labels[i]=str(labels[i]) + " eV"
                    elif labels[i]<1e6:
                        labels[i]=str(labels[i]/1e3) + " keV"
                    elif labels[i] >= 1e6:
                        labels[i]=str(labels[i]/1e6) + " MeV"
                ax3.set_xticklabels(labels)
                ax3.set_xlabel('Energy loss in scatter',fontsize=20)
                ax3.set_ylabel('Frequency',fontsize=20)

                if self.es[self.curr_pos]>=1e6:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e6)+" MeV",fontsize=24)
                elif self.es[self.curr_pos]>=1e3 and self.es[self.curr_pos] < 1e6:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e3) + " keV",fontsize=24)
                elif self.es['self.curr_pos']<1e3:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])) + "eV",fontsize=24)

                fig.canvas.draw()
            elif e.key=="left":
                self.curr_pos-=1
                self.curr_pos%=len(self.es)
                ax1.cla()
                ax2.cla()
                ax3.cla()

                data=dist[dist['SourceEnergy']==self.es[self.curr_pos]]
                tbins=10**np.linspace(-5,5,200)
                ebins=10**np.linspace(np.log10(min(data['EnergyChange'])),np.log10(max(data['EnergyChange'])),200)

                hist=ax1.hist2d(data['TSteps'],data['EnergyChange'],bins=[tbins,ebins],norm=colors.LogNorm())
                ax1.set_xlabel('Time between scatter / ns',fontsize=20)
                ax1.set_ylabel('Energy loss in scatter',fontsize=20)
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                ax1.set_ylim(top=10e7)
                labels=ax1.get_yticks().tolist()
                for i in range(len(labels)):
                    if labels[i]<1000:
                        labels[i]=str(labels[i]) + " eV"
                    elif labels[i]<1e6:
                        labels[i]=str(labels[i]/1e3) + " keV"
                    elif labels[i] >= 1e6:
                        labels[i]=str(labels[i]/1e6) + " MeV"
                ax1.set_yticklabels(labels)
                ax1.set_xlim(1e-5,1e5)
                ax1.fill_between(tbins,0,detect,alpha=0.3,label='Not detectable ($\\textless$ 480 keV)')
                ax1.plot(us,ebins,label='1 $\mu$s')
                ax1.plot(ns,ebins,label='1 ns')
                ax1.plot(tbins,thermal,label='Thermal neutron energy (0.025 eV)')
                ax1.legend(loc='lower left')
                cbar.set_clim(vmin=0,vmax=np.max(hist[0]))
                cbar.draw_all()

                visible=data['EnergyChange']>4.8e5

                ax2.hist(data['TSteps'],bins=tbins,color='darkblue',alpha=0.3,label='All events')
                ax2.hist(data['TSteps'][visible],bins=tbins,color='purple',alpha=0.8,label='Detectable events')
                ax2.hist(data['TSteps'][data['capindet']],bins=tbins,color='purple',alpha=0.5,label='Captured')
                ax2.set_xscale('log')
                ax2.set_xlabel('Time between scatter / ns',fontsize=20)
                ax2.set_ylabel('Frequency',fontsize=20)
                ax2.legend(loc='lower left')

                ax3.hist(data['EnergyChange'],bins=ebins,color='darkblue',alpha=0.3,label='All events')
                ax3.hist(data['EnergyChange'][visible],bins=ebins,color='purple',alpha=0.8,label='Detectable events')
                ax3.hist(data['EnergyChange'][data['capindet']],bins=ebins,color='purple',alpha=0.5,label='Captured')
                ax3.set_xscale('log')
                labels=ax3.get_xticks().tolist()
                for i in range(len(labels)):
                    if labels[i]<1000:
                        labels[i]=str(labels[i]) + " eV"
                    elif labels[i]<1e6:
                        labels[i]=str(labels[i]/1e3) + " keV"
                    elif labels[i] >= 1e6:
                        labels[i]=str(labels[i]/1e6) + " MeV"
                ax3.set_xticklabels(labels)
                ax3.set_xlabel('Energy loss in scatter',fontsize=20)
                ax3.set_ylabel('Frequency',fontsize=20)


                if self.es[self.curr_pos]>=1e6:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e6)+" MeV",fontsize=24)
                elif self.es[self.curr_pos]>=1e3 and self.es[self.curr_pos] < 1e6:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e3) + " keV",fontsize=24)
                elif self.es['self.curr_pos']<1e3:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])) + "eV",fontsize=24)

                fig.canvas.draw()
            else:
                return
        
        fig.canvas.mpl_connect("key_press_event",key_event)

        data=dist[dist['SourceEnergy']==self.es[self.curr_pos]]
        tbins=10**np.linspace(-5,5,200)
        ebins=10**np.linspace(np.log10(min(data['EnergyChange'])),np.log10(max(data['EnergyChange'])),200)

        hist=ax1.hist2d(data['TSteps'],data['EnergyChange'],bins=[tbins,ebins],norm=colors.LogNorm())
        ax1.set_xlabel('Time between scatter / ns',fontsize=20)
        ax1.set_ylabel('Energy loss in scatter',fontsize=20)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylim(top=10e7)
        labels=ax1.get_yticks().tolist()
        for i in range(len(labels)):
            if labels[i]<1000:
                labels[i]=str(labels[i]) + " eV"
            elif labels[i]<1e6:
                labels[i]=str(labels[i]/1e3) + " keV"
            elif labels[i] >= 1e6:
                labels[i]=str(labels[i]/1e6) + " MeV"
        ax1.set_yticklabels(labels)
        ax1.set_xlim(1e-5,1e5)
        ax1.fill_between(tbins,0,detect,alpha=0.3,label='Not detectable ($\\textless$ 480 keV)')
        ax1.plot(us,ebins,label='1 $\mu$s')
        ax1.plot(ns,ebins,label='1 ns')
        ax1.plot(tbins,thermal,label='Thermal neutron energy (0.025 eV)')
        ax1.legend(loc='lower left')
        cbar=plt.colorbar(hist[3],ax=ax1)
        cbar.ax.tick_params(labelsize=20)

        visible=data['EnergyChange']>4.8e5

        ax2.hist(data['TSteps'],bins=tbins,color='darkblue',alpha=0.3,label='All events')
        ax2.hist(data['TSteps'][visible],bins=tbins,color='purple',alpha=0.8,label='Detectable events')
        ax2.hist(data['TSteps'][data['capindet']],bins=tbins,color='purple',alpha=0.5,label='Captured')
        ax2.set_xscale('log')
        ax2.set_xlabel('Time between scatter / ns',fontsize=20)
        ax2.set_ylabel('Frequency',fontsize=20)
        ax2.legend(loc='lower left')
        
        ax3.hist(data['EnergyChange'],bins=ebins,color='darkblue',alpha=0.3,label='All events')
        ax3.hist(data['EnergyChange'][visible],bins=ebins,color='purple',alpha=0.8,label='Detectable events')
        ax3.hist(data['EnergyChange'][data['capindet']],bins=ebins,color='purple',alpha=0.5,label='Captured')
        ax3.set_xscale('log')
        labels=ax3.get_xticks().tolist()
        for i in range(len(labels)):
            if labels[i]<1000:
                labels[i]=str(labels[i]) + " eV"
            elif labels[i]<1e6:
                labels[i]=str(labels[i]/1e3) + " keV"
            elif labels[i] >= 1e6:
                labels[i]=str(labels[i]/1e6) + " MeV"
        ax3.set_xticklabels(labels)
        ax3.set_xlabel('Energy loss in scatter',fontsize=20)
        ax3.set_ylabel('Frequency',fontsize=20)
               


        if self.es[self.curr_pos]>=1e6:
            fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e6)+" MeV",fontsize=24)
        elif self.es[self.curr_pos]>=1e3 and self.es[self.curr_pos] < 1e6:
            fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e3) + " keV",fontsize=24)
        elif self.es['self.curr_pos']<1e3:
            fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])) + "eV",fontsize=24)

        plt.show()

    def ScatterTD(self):
        xpos=self.df['XPosition'].explode()
        ypos=self.df['YPosition'].explode()
        zpos=self.df['ZPosition'].explode()
        indet=(xpos>self.xmin) & (xpos<self.xmax) & (ypos>self.ymin) & (ypos<self.ymax) & (zpos>self.zmin) & (zpos<self.zmax)
        xpos,ypos,zpos=None,None,None

        dist=pd.concat((self.df.loc[:,['TSteps','SourceEnergy']].explode()[indet],self.df['StepLength'].explode()[indet]),axis=1)

        fig=plt.figure(figsize=(19.2,10.8))
        gs=fig.add_gridspec(2,4)
        ax1=fig.add_subplot(gs[:,:2])
        ax2=fig.add_subplot(gs[0,2:])
        ax3=fig.add_subplot(gs[1,2:])

        self.curr_pos=0
        us=np.full(200,1000)
        ns=np.full(200,1)

        def key_event(e):
            if e.key=="right":
                self.curr_pos+=1
                self.curr_pos%=len(self.es)
                ax1.cla()
                ax2.cla()
                ax3.cla()

                data=dist[dist['SourceEnergy']==self.es[self.curr_pos]]
                tbins=10**np.linspace(np.log10(min(data['TSteps'])),np.log10(max(data['TSteps'])),200)
                dbins=10**np.linspace(np.log10(min(data['StepLength'])),np.log10(max(data['StepLength'])),200)

                hist=ax1.hist2d(data['TSteps'],data['StepLength'],bins=[tbins,dbins],norm=colors.LogNorm())
                ax1.set_xlabel('Time between scatter / ns',fontsize=20)
                ax1.set_ylabel('Distance travelled between scatter / mm',fontsize=20)
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                #ax1.set_ylim(top=10e7)
                ax1.set_xlim(1e-5,1e5)
                ax1.plot(us,dbins,label='1 $\mu$s')
                ax1.plot(ns,dbins,label='1 ns')
                ax1.legend(loc='lower left')
                cbar.set_clim(vmin=0,vmax=np.max(hist[0]))
                cbar.draw_all()

                ax2.hist(data['TSteps'],bins=tbins)
                ax2.set_xscale('log')
                ax2.set_xlabel('Time between scatter / ns',fontsize=20)
                ax2.set_ylabel('Frequency',fontsize=20)

                ax3.hist(data['StepLength'],bins=dbins)
                ax3.set_xscale('log')
                ax3.set_xlabel('Distance travelled between scatter / mm',fontsize=20)
                ax3.set_ylabel('Frequency',fontsize=20)

                if self.es[self.curr_pos]>=1e6:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e6)+" MeV",fontsize=24)
                elif self.es[self.curr_pos]>=1e3 and self.es[self.curr_pos] < 1e6:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e3) + " keV",fontsize=24)
                elif self.es['self.curr_pos']<1e3:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])) + "eV",fontsize=24)

                fig.canvas.draw()
            elif e.key=="left":
                self.curr_pos-=1
                self.curr_pos%=len(self.es)
                ax1.cla()
                ax2.cla()
                ax3.cla()

                data=dist[dist['SourceEnergy']==self.es[self.curr_pos]]
                tbins=10**np.linspace(np.log10(min(data['TSteps'])),np.log10(max(data['TSteps'])),200)
                dbins=10**np.linspace(np.log10(min(data['StepLength'])),np.log10(max(data['StepLength'])),200)

                hist=ax1.hist2d(data['TSteps'],data['StepLength'],bins=[tbins,dbins],norm=colors.LogNorm())
                ax1.set_xlabel('Time between scatter / ns',fontsize=20)
                ax1.set_ylabel('Distance travelled between scatter / mm',fontsize=20)
                ax1.set_xscale('log')
                ax1.set_yscale('log')
                #ax1.set_ylim(top=10e7)
                ax1.set_xlim(1e-5,1e5)
                ax1.plot(us,dbins,label='1 $\mu$s')
                ax1.plot(ns,dbins,label='1 ns')
                ax1.legend(loc='lower left')
                cbar.set_clim(vmin=0,vmax=np.max(hist[0]))
                cbar.draw_all()

                ax2.hist(data['TSteps'],bins=tbins)
                ax2.set_xscale('log')
                ax2.set_xlabel('Time between scatter / ns',fontsize=20)
                ax2.set_ylabel('Frequency',fontsize=20)

                ax3.hist(data['StepLength'],bins=dbins)
                ax3.set_xscale('log')
                ax3.set_xlabel('Distance travelled between scatter / mm',fontsize=20)
                ax3.set_ylabel('Frequency',fontsize=20)

                if self.es[self.curr_pos]>=1e6:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e6)+" MeV",fontsize=24)
                elif self.es[self.curr_pos]>=1e3 and self.es[self.curr_pos] < 1e6:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e3) + " keV",fontsize=24)
                elif self.es['self.curr_pos']<1e3:
                    fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])) + "eV",fontsize=24)

                fig.canvas.draw()
            else:
                return
        
        fig.canvas.mpl_connect("key_press_event",key_event)

        data=dist[dist['SourceEnergy']==self.es[self.curr_pos]]
        tbins=10**np.linspace(np.log10(min(data['TSteps'])),np.log10(max(data['TSteps'])),200)
        dbins=10**np.linspace(np.log10(min(data['StepLength'])),np.log10(max(data['StepLength'])),200)

        hist=ax1.hist2d(data['TSteps'],data['StepLength'],bins=[tbins,dbins],norm=colors.LogNorm())
        ax1.set_xlabel('Time between scatter / ns',fontsize=20)
        ax1.set_ylabel('Distance travelled between scatter / mm',fontsize=20)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        #ax1.set_ylim(top=10e7)
        ax1.set_xlim(1e-5,1e5)
        ax1.plot(us,dbins,label='1 $\mu$s')
        ax1.plot(ns,dbins,label='1 ns')
        ax1.legend(loc='lower left')
        cbar=plt.colorbar(hist[3],ax=ax1)
        cbar.ax.tick_params(labelsize=20)

        ax2.hist(data['TSteps'],bins=tbins)
        ax2.set_xscale('log')
        ax2.set_xlabel('Time between scatter / ns',fontsize=20)
        ax2.set_ylabel('Frequency',fontsize=20)

        ax3.hist(data['StepLength'],bins=dbins)
        ax3.set_xscale('log')
        ax3.set_xlabel('Distance travelled between scatter / mm',fontsize=20)
        ax3.set_ylabel('Frequency',fontsize=20)

        if self.es[self.curr_pos]>=1e6:
            fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e6)+" MeV",fontsize=24)
        elif self.es[self.curr_pos]>=1e3 and self.es[self.curr_pos] < 1e6:
            fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])/1e3) + " keV",fontsize=24)
        elif self.es['self.curr_pos']<1e3:
            fig.suptitle("Monoenergetic " + str(int(self.es[self.curr_pos])) + "eV",fontsize=24)

        plt.show()

    def CapEfficiency(self,log=True,sheet="/home/nr1315/Downloads/ICRP74_neutron_dose equivalent curves.xlsx"):
        effs=[]
        effserr=[]
        effs_0=[]
        effserr_0=[]
        effs_1=[]
        effserr_1=[]
        effs_2=[]
        effserr_2=[]
        scatteffs=[]
        scatteffserr=[]
        scatt_1=[]
        scatterr_1=[]
        scatt_2=[]
        scatterr_2=[]
        for e in self.es:
            data=self.df[self.df['SourceEnergy']==e]
            xpos=data['XPosition'].explode()
            ypos=data['YPosition'].explode()
            zpos=data['ZPosition'].explode()
            capindet=(xpos>self.xmin)&(xpos<self.xmax)&(ypos>self.ymin)&(ypos<self.ymax)&(zpos>self.zmin)&(zpos<self.zmax)
            xpos,ypos,zpos=None,None,None

            scatteffs.append(100*(data['NScattOverThresh']>0).sum()/len(data))
            scatteffserr.append(100*np.sqrt((data['NScattOverThresh']>0).sum())/len(data))

            scatt_1.append(100*(data['NScattOverThresh']==1).sum()/len(data))
            scatterr_1.append(100*np.sqrt((data['NScattOverThresh']==1).sum())/len(data))
            
            scatt_2.append(100*(data['NScattOverThresh']==2).sum()/len(data))
            scatterr_2.append(100*np.sqrt((data['NScattOverThresh']==2).sum())/len(data))

            count=data['capindet'].sum()
            effs.append(100*count/len(data))
            effserr.append(100*np.sqrt(count)/len(data))

            capscatt=data['NScattOverThresh'][data['capindet']]
            effs_1.append(100*(capscatt==1).sum()/len(data))
            effserr_1.append(100*np.sqrt((capscatt==1).sum())/len(data))
            effs_2.append(100*(capscatt==2).sum()/len(data))
            effserr_2.append(100*np.sqrt((capscatt==2).sum())/len(data))

        effserr_1=[np.nan if isinf(x) else x for x in effserr_1]
        effserr_2=[np.nan if isinf(x) else x for x in effserr_2]

        df=pd.read_excel(sheet)
        data=df.iloc[2:,:2]


        fig=plt.figure(figsize=(19.2,10.8))
        ax=fig.add_subplot(111)
        energies=np.array(self.es)/1e3

        ax.plot(energies,effs,c='xkcd:tangerine',label='All captured events',linestyle='--')
        ax.errorbar(energies,effs,yerr=effserr,fmt='none',ecolor='xkcd:tangerine')
                

        ax.plot(energies,effs_1,c='r',label='1 scatter + capture')
        ax.errorbar(energies,effs_1,yerr=effserr_1,fmt='none',ecolor='r')

        ax.plot(energies,effs_2,c='xkcd:dark red',label='2 scatters + capture')
        ax.errorbar(energies,effs_2,yerr=effserr_2,fmt='none',ecolor='xkcd:dark red')

        ax.plot(energies,scatteffs,c='xkcd:light blue',label="All scatter events",linestyle='--')
        ax.errorbar(energies,scatteffs,yerr=scatteffserr,fmt='none',ecolor='xkcd:light blue')

        ax.plot(energies,scatt_1,c='b',label='1 detectable scatter')
        ax.errorbar(energies,scatt_1,yerr=scatterr_1,fmt='none',color='b')

        ax.plot(energies,scatt_2,c='xkcd:deep blue',label='2 detectable scatters')
        ax.errorbar(energies,scatt_2,yerr=scatterr_2,fmt='none',color='xkcd:deep blue')

        ax.set_xlabel('Energy / keV',fontsize=20)
        ax.set_ylabel('Detection efficiency / $\%$',fontsize=20)
        if log:
            ax.set_xscale('log')
        ax.set_title('Simulated detection efficiency of scatter and capture for monoenergetic sources',fontsize=24)
        ax.legend(markerscale=2,scatterpoints=3,loc='upper left')

        ax2=ax.twinx()
        ax2.plot(data.iloc[:,0]*1000,data.iloc[:,1],c='g')
        ax2.set_ylabel('$H^*(10)$ / Sv',fontsize=20,c='g')
        ax.set_xlim(left=10,right=20e3)
        ax2.tick_params(axis='y',labelcolor='g')

        plt.show()

    def DistancePlots(self,first=False):
        xpos=self.df['XPosition'].explode()
        ypos=self.df['YPosition'].explode()
        zpos=self.df['ZPosition'].explode()
        indet=(xpos>self.xmin)&(xpos<self.xmax)&(ypos>self.ymin)&(ypos<self.ymax)&(zpos>self.zmin)&(zpos<self.zmax)
        xpos=xpos[indet]
        ypos=ypos[indet]
        zpos=zpos[indet]

        dist=self.df.loc[:,['StepTime','CaptureTime','CapPosX','CapPosY','CapPosZ','capindet','NScattOverThresh','SourceEnergy']].explode('StepTime')[indet]
        energy=self.df['EnergyChange'].explode()[indet]
        dist=pd.concat((dist,xpos,ypos,zpos,energy),axis=1)
        dist=dist[dist['capindet']]
        dist=dist[dist['EnergyChange']<=-4.8e5]
        if first:
            ind=0
        else:
            ind=-1
        inds=[np.where(dist.index==s)[0][ind] for s in np.unique(dist.index)]
        dist=dist.iloc[inds]

        dist['ScattToCap']=((dist['CapPosX']-dist['XPosition'])**2+(dist['CapPosY']-dist['YPosition'])**2+(dist['CapPosZ']-dist['ZPosition'])**2)**(1/2)

        dbins=np.linspace(0,500,100)
        fig=plt.figure(figsize=(19.2,10.8))
        ax=fig.add_subplot(111)
        self.curr_pos=0
        
        def key_event(e):
            if e.key=="right":
                self.curr_pos+=1
                self.curr_pos%=len(self.es)
                ax.cla()

                data=dist[dist['SourceEnergy']==self.es[self.curr_pos]]
            
                hist=ax.hist(data['ScattToCap'],bins=dbins)
                ax.set_xlabel('Distance from scatter to capture / mm',fontsize=20)
                ax.set_ylabel('Frequency',fontsize=20)
                if self.es[self.curr_pos]>=1e6:
                    ax.set_title("Monoenergetic " + str(int(self.es[self.curr_pos])/1e6)+" MeV",fontsize=24)
                elif self.es[self.curr_pos]>=1e3 and self.es[self.curr_pos] < 1e6:
                    ax.set_title("Monoenergetic " + str(int(self.es[self.curr_pos])/1e3) + " keV",fontsize=24)
                elif self.es['self.curr_pos']<1e3:
                    ax.set_title("Monoenergetic " + str(int(self.es[self.curr_pos])) + "eV",fontsize=24)

                fig.canvas.draw()
            if e.key=="left":
                self.curr_pos-=1
                self.curr_pos%=len(self.es)
                ax.cla()
                data=dist[dist['SourceEnergy']==self.es[self.curr_pos]]

                hist=ax.hist(data['ScattToCap'],bins=dbins)
                ax.set_xlabel('Distance from scatter to capture / mm',fontsize=20)
                ax.set_ylabel('Frequency',fontsize=20)
                if self.es[self.curr_pos]>=1e6:
                    ax.set_title("Monoenergetic " + str(int(self.es[self.curr_pos])/1e6)+" MeV",fontsize=24)
                elif self.es[self.curr_pos]>=1e3 and self.es[self.curr_pos] < 1e6:
                    ax.set_title("Monoenergetic " + str(int(self.es[self.curr_pos])/1e3) + " keV",fontsize=24)
                elif self.es['self.curr_pos']<1e3:
                    ax.set_title("Monoenergetic " + str(int(self.es[self.curr_pos])) + "eV",fontsize=24)

                fig.canvas.draw()
        fig.canvas.mpl_connect("key_press_event",key_event)

        data=dist[dist['SourceEnergy']==self.es[self.curr_pos]]

        ax.hist(data['ScattToCap'],bins=dbins)
        ax.set_xlabel('Distance from scatter to capture / mm',fontsize=20)
        ax.set_ylabel('Frequency',fontsize=20)
        if self.es[self.curr_pos]>=1e6:
            ax.set_title("Monoenergetic " + str(int(self.es[self.curr_pos])/1e6)+" MeV",fontsize=24)
        elif self.es[self.curr_pos]>=1e3 and self.es[self.curr_pos] < 1e6:
            ax.set_title("Monoenergetic " + str(int(self.es[self.curr_pos])/1e3) + " keV",fontsize=24)
        elif self.es['self.curr_pos']<1e3:
            ax.set_title("Monoenergetic " + str(int(self.es[self.curr_pos])) + "eV",fontsize=24)

        plt.show()

    def CapCubeDist(self):
        self.curr_pos=0
        data=self.df[self.df['capindet']].loc[:,['cubex','SourceEnergy']].explode('cubex')
        inds=[np.where(data.index==s)[0][-1] for s in np.unique(data.index)]
        es=self.es
        dd=pd.concat((self.df['cubex'][self.df['capindet']].explode().iloc[inds],self.df['cubey'][self.df['capindet']].explode().iloc[inds],self.df.loc[:,['cubez','SourceEnergy']].explode('cubez')[self.df['capindet']].iloc[inds]),axis=1)

        dd['cubeid']=dd['cubex']-1 + 4*(dd['cubey']-1) + 16*(dd['cubez'] - 1)
        fig=plt.figure()
        ax=fig.add_subplot(111)
        fig.tight_layout()
        self.curr_pos=0
        def key_event(e,dd=dd):
            if e.key=="right":
                self.curr_pos+=1
                self.curr_pos%=len(self.es)

                ax.cla()
                y,binedges,patch=ax.hist(dd['cubeid'][dd['SourceEnergy']==es[self.curr_pos]],bins=64,range=(0,63),ec='b')
                bincenters=0.5*(binedges[1:]+binedges[:-1])
                err=np.sqrt(y)
                ax.errorbar(bincenters,y,yerr=err,fmt='none',c='lightblue')
                ax.set_xlabel('Cube ID',fontsize=20)
                ax.set_ylabel('Frequency',fontsize=20)
                ax.set_ylim(0,100)
                if es[self.curr_pos]>=1e6:
                    ax.set_title('Monoenergetic ' + str(int(es[self.curr_pos]/1e6)) + " MeV",fontsize=24)
                elif es[self.curr_pos]>=1e3 and es[self.curr_pos]<1e6:
                    ax.set_title('Monoenergetic ' + str(int(es[self.curr_pos]/1e3)) + " keV",fontsize=24)
                elif es[self.curr_pos]<1e3:
                    ax.set_title("Monoenergetic " + str(int(es[self.curr_pos])) + " eV",fontsize=24)
                fig.canvas.draw()
            elif e.key=="left":
                self.curr_pos-=1
                self.curr_pos%=len(es)
                ax.cla()
                y,binedges,patch=ax.hist(dd['cubeid'][dd['SourceEnergy']==es[self.curr_pos]],bins=64,range=(0,63),ec='b')
                bincenters=0.5*(binedges[1:]+binedges[:-1])
                err=np.sqrt(y)
                ax.errorbar(bincenters,y,yerr=err,fmt='none',c='lightblue')

                ax.set_xlabel('Cube ID',fontsize=20)
                ax.set_ylabel('Frequency',fontsize=20)
                ax.set_ylim(0,100)
                if es[self.curr_pos]>=1e6:
                    ax.set_title('Monoenergetic ' + str(int(es[self.curr_pos]/1e6)) + " MeV",fontsize=24)
                elif es[self.curr_pos]>=1e3 and es[self.curr_pos]<1e6:
                    ax.set_title('Monoenergetic ' + str(int(es[self.curr_pos]/1e3)) + " keV",fontsize=24)
                elif es[self.curr_pos]<1e3:
                    ax.set_title("Monoenergetic " + str(int(es[self.curr_pos])) + " eV",fontsize=24)
                fig.canvas.draw()
            else:
                return

        fig.canvas.mpl_connect("key_press_event",key_event)
        y,binedges,path=ax.hist(dd['cubeid'][dd['SourceEnergy']==es[self.curr_pos]],bins=64,range=(0,63),ec='b')
        bincenters=0.5*(binedges[1:]+binedges[:-1])
        err=np.sqrt(y)
        ax.errorbar(bincenters,y,yerr=err,fmt='none',c='lightblue')

        ax.set_xlabel('Cube ID',fontsize=20)
        ax.set_ylabel('Frequency',fontsize=20)
        ax.set_ylim(0,100)
        if es[0]>=1e6:
            ax.set_title('Monoenergetic ' + str(int(es[0]/1e6)) + " MeV",fontsize=24)
        elif es[0]>=1e3 and es[0]<1e6:
            ax.set_title('Monoenergetic ' + str(int(es[0]/1e3)) + " keV",fontsize=24)
        elif es[0]<1e3:
            ax.set_title("Monoenergetic " + str(int(es[0])) + " eV",fontsize=24)
            
        plt.show()


class MonoDistViewer2():
    def __init__(self,folder,loadfile=True):
        if loadfile:
            self.df=pd.read_hdf(folder+"/monos.h5")
            self.es=np.unique(self.df['SourceEnergy'])
            return

        files=[name for name in os.listdir(folder) if name[-6:]=="kev.h5" or name[-6:]=="MeV.h5" or name[-6:]=="keV.h5"]
        energies={'kev':1000,'keV':1000,'MeV':1e6}
        name=files.pop()
        df=pd.read_hdf(folder+"/"+name)
        ind=df.index
        e=float(name[5:-6])*energies[name[-6:-3]]
        df['SourceEnergy']=[e]*len(ind)
        for file in files:
            data=pd.read_hdf(folder+"/"+file)
            ind=data.index
            e=float(file[5:-6])*energies[file[-6:-3]]
            data['SourceEnergy']=[e]*len(ind)
            df=pd.concat((df,data))
        df.reset_index(inplace=True,drop=True)

        t=pd.concat((df.loc[:,['edep_pvt','CapCube']].explode('edep_pvt'),df['pdg'].explode()),axis=1)
        t['detectable_scatt']=(t['edep_pvt']>480e-3)&((t['pdg']==2112)|(t['pdg']==2212))
        scatts=[]
        for ind in np.unique(t.index):
            scatts.append(t.loc[ind,'detectable_scatt'].sum())
        
        df['NScatt']=scatts

        exp=pd.concat([df['pdg'].explode(),df.loc[:,['cubeid','CapCube']].explode('cubeid')],axis=1)
        ind=[i for i in exp.index]
        unq,counts=np.unique(ind,return_counts=True)
        arr=list(itertools.chain(*[list(range(count)) for count in counts]))
        tups=list(zip(*[ind,arr]))
        mind=pd.MultiIndex.from_tuples(tups)
        exp.set_index(mind,inplace=True)
        lvl1=np.unique(exp.index.get_level_values(0))
        caps=[]
        for i in lvl1:
            eve=exp.loc[i,['pdg','cubeid','CapCube']]
            if len(eve)>1:
                if (eve['pdg'].iloc[-2]==1000020040)&(eve['pdg'].iloc[-1]==1000010030):
                    caps.append(eve['cubeid'].iloc[-1])
                else:
                    caps.append(eve['CapCube'].iloc[-1])
            else:
                caps.append(eve['CapCube'].values[0])
        df['CapCube']=caps

        diffs=[]
        for t in df['StepTime']:
            t_roll=np.roll(np.append(np.diff(t),t[0]),1)
            diffs.append(t_roll)
        df['TSteps']=diffs
        self.df=df
        self.es=np.unique(df['SourceEnergy'])

        df.to_hdf(folder+"/monos.h5",key='monos')
        return

    def CapEfficiency(self,log=True,sheet="/home/nr1315/Downloads/ICRP74_neutron_dose equivalent curves.xlsx",save=False):
        effs=[]
        effserr=[]
        effs_0=[]
        effserr_0=[]
        effs_1=[]
        effserr_1=[]
        effs_2=[]
        effserr_2=[]
        scatteffs=[]
        scatteffserr=[]
        scatt_1=[]
        scatterr_1=[]
        scatt_2=[]
        scatterr_2=[]
        for e in self.es:
            data=self.df[self.df['SourceEnergy']==e]

            scatteffs.append(100*(data['NScatt']>0).sum()/len(data))
            scatteffserr.append(100*np.sqrt((data['NScatt']>0).sum())/len(data))

            scatt_1.append(100*(data['NScatt']==1).sum()/len(data))
            scatterr_1.append(100*np.sqrt((data['NScatt']==1).sum())/len(data))
            
            scatt_2.append(100*(data['NScatt']==2).sum()/len(data))
            scatterr_2.append(100*np.sqrt((data['NScatt']==2).sum())/len(data))

            count=(data['CapCube']>0).sum()
            effs.append(100*count/len(data))
            effserr.append(100*np.sqrt(count)/len(data))

            capscatt=data['NScatt'][data['CapCube']>0]
            effs_1.append(100*(capscatt==1).sum()/len(data))
            effserr_1.append(100*np.sqrt((capscatt==1).sum())/len(data))
            effs_2.append(100*(capscatt==2).sum()/len(data))
            effserr_2.append(100*np.sqrt((capscatt==2).sum())/len(data))

        effserr_1=[np.nan if isinf(x) else x for x in effserr_1]
        effserr_2=[np.nan if isinf(x) else x for x in effserr_2]

        df=pd.read_excel(sheet)
        data=df.iloc[2:,:2]


        fig=plt.figure(figsize=(19.2,10.8))
        ax=fig.add_subplot(111)
        energies=np.array(self.es)/1e3

        ax.plot(energies,effs,c='xkcd:tangerine',label='All captured events',linestyle='--')
        ax.errorbar(energies,effs,yerr=effserr,fmt='none',ecolor='xkcd:tangerine')
                

        ax.plot(energies,effs_1,c='r',label='1 scatter + capture')
        ax.errorbar(energies,effs_1,yerr=effserr_1,fmt='none',ecolor='r')

        ax.plot(energies,effs_2,c='xkcd:dark red',label='2 scatters + capture')
        ax.errorbar(energies,effs_2,yerr=effserr_2,fmt='none',ecolor='xkcd:dark red')

        ax.plot(energies,scatteffs,c='xkcd:light blue',label="All scatter events",linestyle='--')
        ax.errorbar(energies,scatteffs,yerr=scatteffserr,fmt='none',ecolor='xkcd:light blue')

        ax.plot(energies,scatt_1,c='b',label='1 detectable scatter')
        ax.errorbar(energies,scatt_1,yerr=scatterr_1,fmt='none',color='b')

        ax.plot(energies,scatt_2,c='xkcd:deep blue',label='2 detectable scatters')
        ax.errorbar(energies,scatt_2,yerr=scatterr_2,fmt='none',color='xkcd:deep blue')

        ax.set_xlabel('Energy / keV',fontsize=20)
        ax.set_ylabel('Detection efficiency / $\%$',fontsize=20)
        if log:
            ax.set_xscale('log')
        ax.set_title('Simulated detection efficiency of scatter and capture for monoenergetic sources',fontsize=24)
        ax.legend(markerscale=2,scatterpoints=3,loc='upper left')

        ax2=ax.twinx()
        ax2.plot(data.iloc[:,0]*1000,data.iloc[:,1],c='g')
        ax2.set_ylabel('$H^*(10)$ / Sv',fontsize=20,c='g')
        ax.set_xlim(left=10,right=20e3)
        ax2.tick_params(axis='y',labelcolor='g')

        if save:
            s=np.vstack([energies,effs])
            np.save('/home/nr1315/Documents/Project/Simulations/Data/Low_scatter_old/efficiecy.npy',s)


        plt.show()



class NewSimPlotter():
    def __init__(self):
        self.data={}

    def add_data(self,dfile,name):
        df=pd.read_hdf(dfile)
        df.reset_index(inplace=True,drop=True)
        exp=pd.concat([df['pdg'].explode(),df.loc[:,['cubeid','CapCube']].explode('cubeid')],axis=1)
        ind=[i for i in exp.index]
        unq,counts=np.unique(ind,return_counts=True)
        arr=list(itertools.chain(*[list(range(count)) for count in counts]))
        tups=list(zip(*[ind,arr]))
        mind=pd.MultiIndex.from_tuples(tups)
        exp.set_index(mind,inplace=True)
        lvl1=np.unique(exp.index.get_level_values(0))
        caps=[]
        for i in lvl1:
            eve=exp.loc[i,['pdg','cubeid','CapCube']]
            if len(eve)>1:
                if (eve['pdg'].iloc[-2]==1000020040)&(eve['pdg'].iloc[-1]==1000010030):
                    caps.append(eve['cubeid'].iloc[-1])
                else:
                    caps.append(eve['CapCube'].iloc[-1])
            else:
                caps.append(eve['CapCube'].values[0])
        df['CapCube']=caps
        self.data[name]=df

    def CombineDatasets(self,file1,file2,name):
        df1=self.data[file1]
        df2=self.data[file2]
        df=pd.concat([df1,df2],axis=0)
        self.data[name]=df
        self.data.pop(file1,None)
        self.data.pop(file2,None)
    
    def View_3D(self,dfile):
        df=self.data[dfile]
        cubes=df.query('CapCube>0')['CapCube'].values
        z=(cubes/1e4).astype(int)
        x=((cubes-z*1e4)/100).astype(int)
        y=(cubes-z*1e4-x*100).astype(int)
        x-=1
        y-=1
        z-=1

        ze=np.zeros([4,4,4])
        coords=np.argwhere(ze==0)
        cubs=pd.DataFrame(np.array([x,y,z]).T,columns=['X','Y','Z'])
        counts=[]
        for i in coords:
            counts.append(np.count_nonzero((cubs.values==i).all(axis=1)))
        ze[coords[:,0],coords[:,1],coords[:,2]]=counts
        arr=ze
        sc=(50.888-50)/50

        p=ndap.NDArrayPlotter(arr)
        p.set_alpha(0.05)

        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        fig.set_tight_layout(True)

        cmap=cm.viridis
        norm=colors.Normalize(vmin=0,vmax=np.max(arr))
        p.colors=cmap(norm(arr))
        alph=norm(arr)*0.95
        p.alphas=alph
        sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm)
        sm.set_array(arr)


        p.render(azim=-56,elev=25,ax=ax,text=None,labels=True,space=0.5)
        ax.quiver(-0.4,-0.4,-0.4,1,0,0,length=5,arrow_length_ratio=0.05,color='black')
        ax.quiver(-0.4,-0.4,-0.4,0,1,0,length=5,arrow_length_ratio=0.05,color='black')
        ax.quiver(-0.4,-0.4,-0.4,0,0,1,length=5,arrow_length_ratio=0.05,color='black')
        cbar=plt.colorbar(sm,ax=ax)
        cbar.set_label('Event count',rotation=270,fontsize=30,labelpad=30)
        plt.show()
        

    def fit_projection(self,dfile):
        df=self.data[dfile]
        cubes=df.query('CapCube>0')['CapCube'].values
        z=(cubes/1e4).astype(int)
        x=((cubes-z*1e4)/100).astype(int)
        y=(cubes-z*1e4-x*100).astype(int)

        planes=pd.DataFrame(np.array([np.unique(x,return_counts=True)[1],np.unique(y,return_counts=True)[1],np.unique(z,return_counts=True)[1]]).T,columns=['X','Y','Z'])
        
        def en(x,a,b):
            return a*np.exp(-1*x/b)
        
        def ep(x,a,b):
            return a*np.exp(x/b)

        xyz=['X','Y','Z']
        diffs=planes.diff().iloc[1:]
        parameters=np.zeros([3,4])
        for i in range(3):
            if i!=1:
                parameters[i]=np.nan
                continue
            plane=xyz[i]
            if planes.loc[planes[plane]==max(planes[plane])].index[0]==0:
                #if (diffs[plane]<0).all(axis=0):
                ma=max(planes[plane])
                popt,pcov=curve_fit(en,planes.index.values,planes[plane].values,p0=[ma,2],sigma=np.sqrt(planes[plane].values))
                #if pcov[1,1] < 1:
                parameters[i,0],parameters[i,2]=popt[0],popt[1]
                parameters[i,1],parameters[i,3]=pcov[0,0],pcov[1,1]
                    #else:
                    #    parameters[i]=np.nan
                #else:
                #    parameters[i]=np.nan
            elif planes.loc[planes[plane]==max(planes[plane])].index[0]==1:
                parameters[i]=np.nan
                """
                if (diffs[plane].iloc[1:]<0).all():
                    ma=max(planes[plane])
                    popt,pcov=curve_fit(en,planes.index.values[1:],planes[plane].values[1:],p0=[ma,2],sigma=np.sqrt(planes[plane].values[1:]))
                    if pcov[1,1]<1:
                        parameters[i,0],parameters[i,2]=popt[0],popt[1]
                        parameters[i,1],parameters[i,3]=pcov[0,0],pcov[1,1]
                    else:
                        parameters[i,0],parameters[i,1],parameters[i,2],parameters[i,3]=np.nan,np.nan,np.nan,np.nan
                """
            elif planes.loc[planes[plane]==max(planes[plane])].index[0]==2:
                parameters[i]=np.nan
                """
                if (diffs[plane].iloc[:-1]>0).all():
                    ma=max(planes[plane])
                    popt,pcov=curve_fit(ep,planes.index.values[:-1],planes[plane].values[:-1],p0=[ma,2],sigma=np.sqrt(planes[plane].values[:-1]))
                    if pcov[1,1]<1:
                        parameters[i,0],parameters[i,2]=popt[0],popt[1]
                        parameters[i,1],parameters[i,3]=pcov[0,0],pcov[1,1]
                    else:
                        parameters[i,0],parameters[i,1],parameters[i,2],parameters[i,3]=np.nan,np.nan,np.nan,np.nan
                
                """
            elif planes.loc[planes[plane]==max(planes[plane])].index[0]==3:
                ma=max(planes[plane])
                popt,pcov=curve_fit(ep,planes.index.values,planes[plane].values,p0=[ma,2],sigma=np.sqrt(planes[plane].values))
                    #if pcov[1,1]<1:
                parameters[i,0],parameters[i,2]=popt[0],popt[1]
                parameters[i,1],parameters[i,3]=pcov[0,0],pcov[1,1]
                    #else:
                     #   parameters[i]=np.nan
                #else:
                #    parameters[i]=np.nan

        space=np.linspace(-0.2,3.2,100)
        fig,ax=plt.subplots(1,3,figsize=(30,10))
        ax[0].bar(planes.index,planes['X'].values,width=1,edgecolor='b',alpha=0.5,yerr=np.sqrt(planes['X'].values),color='xkcd:cerulean')
        if not np.isnan(parameters[0,0]):
            if np.argmax(planes['X'])<=1:
                func=en
                ax[0].plot(space,func(space,parameters[0,0],parameters[0,2]),c='r',label=r'$\lambda$ = {}'.format(-1*round(parameters[0,2],2)))   
            else:
                func=ep
                ax[0].plot(space,func(space,parameters[0,0],parameters[0,2]),c='r',label=r'$\lambda$ = {}'.format(round(parameters[0,2],2)))   
        
        ax[1].bar(planes.index,planes['Y'].values,width=1,edgecolor='b',alpha=0.5,yerr=np.sqrt(planes['Y'].values),color='xkcd:cerulean')
        if not np.isnan(parameters[1,0]):
            if np.argmax(planes['Y'])<=1:
                func=en
                ax[1].plot(space,func(space,parameters[1,0],parameters[1,2]),c='r',label=r'$\lambda$ = {}'.format(-1*round(parameters[1,2],2)))
            else:
                func=ep
                ax[1].plot(space,func(space,parameters[1,0],parameters[1,2]),c='r',label=r'$\lambda$ = {}'.format(round(parameters[1,2],2)))

        ax[2].bar(planes.index,planes['Z'].values,width=1,edgecolor='b',alpha=0.5,yerr=np.sqrt(planes['Z'].values),color='xkcd:cerulean')
        if not np.isnan(parameters[2,0]):
            if np.argmax(planes['Z'])<=1:
                func=en
                ax[2].plot(space,func(space,parameters[2,0],parameters[2,2]),c='r',label=r'$\lambda$ = {}'.format(-1*round(parameters[2,2],2)))
            else:
                func=ep
                ax[2].plot(space,func(space,parameters[2,0],parameters[2,2]),c='r',label=r'$\lambda$ = {}'.format(round(parameters[2,2],2)))
            
            

        pl=['X planes','Y planes','Z planes']
        for i in range(3):
            ax[i].set_xlabel('Plane number',fontsize=26)
            ax[i].set_ylabel('Background corrected count',fontsize=26)
            ax[i].set_title(pl[i],fontsize=30)
            if not np.isnan(parameters[i,0]):
                ax[i].legend(loc='upper right',fontsize=36)
        
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[i].tick_params(labelsize=26)
            
        #fig.set_tight_layout(True)
        fig.suptitle(dfile,fontsize=30)
        #fig.subplots_adjust(left=0.05,bottom=0.09,right=0.99,top=0.31,wspace=0.17,hspace=0.2)
        plt.show()
        return parameters