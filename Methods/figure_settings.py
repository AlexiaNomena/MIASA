import matplotlib
import matplotlib.pyplot as pl
import numpy as np
import scipy as scp
import pdb
import seaborn as sns
from .Core.Tensor_Laws import convert_spherical
#from Core.Lower_dim import sklearn_embedding
from copy import copy

#### Visualisation ###  
def PreFig(xsize = 12, ysize = 12):
    '''
    @brief: customize figure parameters
    '''
    matplotlib.rc('xtick', labelsize=xsize) 
    matplotlib.rc('ytick', labelsize=ysize)


def Separation_axis(ax, xy_rows, xy_cols, outliers, out = 1.2, lims = True, col = "black"):
    if outliers is not None:
        outliers_rows, outliers_cols = outliers  
        if outliers_rows*(not outliers_cols):
            print("remove outliers of columns variable")
            xmin, xmax = out*np.amin(xy_rows[:, 0]), out*np.amax(xy_rows[:, 0])
            ymin, ymax = out*np.amin(xy_rows[:, 1]), out*np.amax(xy_rows[:, 1])
            
        elif outliers_cols*(not outliers_rows):
            print("remove outliers of rows variable")
            xmin, xmax = out*np.amin(xy_cols[:, 0]), out*np.amax(xy_cols[:, 0])
            ymin, ymax = out*np.amin(xy_cols[:, 1]), out*np.amax(xy_cols[:, 1])
            
        elif (not outliers_rows)*(not outliers_cols):
            print("remove outliers of both rows and columns variables")
            out = 0.90
            xmin1, xmax1 = out*np.amin(xy_rows[:, 0]), out*np.amax(xy_rows[:, 0])
            ymin1, ymax1 = out*np.amin(xy_rows[:, 1]), out*np.amax(xy_rows[:, 1])
            
            xmin2, xmax2 = out*np.amin(xy_cols[:, 0]), out*np.amax(xy_cols[:, 0])
            ymin2, ymax2 = out*np.amin(xy_cols[:, 1]), out*np.amax(xy_cols[:, 1])
            
            xmin, xmax = min(xmin1, xmin2), min(xmax1, xmax2)
            ymin, ymax = min(ymin1, ymin2), min(ymax1, ymax2)
            
        else:
            print("plot all datapoints")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
    else:
        print("plot all datapoints")
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
    
      
    if xmin<0:
        pl.plot(np.linspace(xmin, xmax, 5), np.zeros(5), "--", color = col, linewidth = 0.5)
    if ymin<0:
        pl.plot(np.zeros(5), np.linspace(ymin, ymax, 5), "--", color = col, linewidth = 0.5)        
    
    if lims:
        pl.xlim((xmin, xmax))
        pl.ylim((ymin, ymax))
    
    return ax


def OneAnnotation(ax, lab, coords, col_val, xl=5, yl=5, arrow = False, fontsize = 12, alpha = 0.5):
    if arrow:
        ax.annotate("%s"%lab, xy=coords, 
                xytext= (xl, yl), textcoords='offset points', ha='center', va='bottom',
                #bbox=dict(boxstyle='round,pad=0.2', fc=col_val, alpha=alpha),
                bbox=dict(boxstyle='circle', fc=col_val, alpha=alpha),
                arrowprops=dict(arrowstyle='->', color = "black"),  #connectionstyle='arc3,rad=0.5'),
                color= "black",
                fontsize = fontsize # 6
                 )
    else:
         ax.annotate("%s"%lab, xy=coords, 
                xytext= (xl, yl), textcoords='offset points', ha='center', va='bottom',
                #bbox=dict(boxstyle='round,pad=0.2', fc=col_val, alpha=alpha),
                bbox=dict(boxstyle='circle', fc=col_val, alpha=alpha),
                color= "black",
                fontsize = fontsize # 6
                 )
    return ax


def Annotate(ax, rows_to_Annot, cols_to_Annot, Label_rows, Label_cols, xy_rows, xy_cols, col = ("green", "pink"), arrow = False):
    '''
    @brief : plot text annotations 
    @params: see function CA 
    '''
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    pdist_n = scp.spatial.distance.pdist(np.concatenate((xy_rows,xy_cols), axis = 0))
    pdist_n[np.isnan(pdist_n)] = 10000000
    pdist_n[~np.isfinite(pdist_n)] = 10000000
    
    pdist = scp.spatial.distance.squareform(pdist_n)
    
    
    if rows_to_Annot is not None: #and Label_rows is not None:
        #pdist_n = scp.spatial.distance.pdist(xy_rows)
        #pdist_n[np.isnan(pdist_n)] = 10000000
        #pdist_n[~np.isfinite(pdist_n)] = 10000000
        
        #pdist = scp.spatial.distance.squareform(pdist_n)
        
        l_special = []
        sup = 0
        f = 1.75
        for j in rows_to_Annot:
            if (np.sum(pdist[j, (j+1):] <= 0.15*np.mean(pdist_n)) != 0)*(j not in l_special)*(j != xy_rows.shape[0]-1):
                xl = 0#20*(f)*np.cos(sup) #0
                yl = -20 #20*(f)*np.sin(sup) #-10#
                l_special.append(j)
            else:
                xl = 0#20*(f)*np.cos(sup) #0
                yl = 20#20*(f)*np.sin(sup) #5#
            
            f += 0.05
            sup += 2*np.pi/len(rows_to_Annot)
            
            try:
                col_j = col[0][Label_rows[j]]
            except:
                try:
                    col_j = col[0]
                except:
                    print("Column categories are pink because you haven't entered an appropriate col parameter: it's either a tuple of 2 colors (one for all the row sand ones for all the cols) or a tuple of 2 dictionaries with labels as keys (one for all the rows and one for all the cols) ")
                    col_j = "green"
            
            try:
                ax = OneAnnotation(ax, int(Label_rows[j]), xy_rows[j, :], col_j, xl, yl, arrow = arrow)
            except:
                ax = OneAnnotation(ax, Label_rows[j], xy_rows[j, :], col_j, xl, yl, arrow = arrow)
        
    
    if cols_to_Annot is not None: #and Label_cols is not None:
        #pdist_n = scp.spatial.distance.pdist(xy_cols)
        #pdist_n[np.isnan(pdist_n)] = 10000000
        #pdist_n[~np.isfinite(pdist_n)] = 10000000
        
        #pdist = scp.spatial.distance.squareform(pdist_n)
        
        
        l_special = []
        for j in cols_to_Annot: 
            try:
                col_j = col[1][Label_cols[j]]
            except:
                try:
                    col_j = col[1]
                except:
                    print("Column categories are pink because you haven't entered an appropriate col parameter: it's either a tuple of 2 colors (one for all the row sand ones for all the cols) or a tuple of 2 dictionaries with labels as keys (one for all the rows and one for all the cols) ")
                    col_j = "pink"
    
            if (np.sum(pdist[j, (j+1):] <= 0.20*np.mean(pdist_n)) != 0)*(j not in l_special)*(j != xy_cols.shape[0]-1):
                xl =  0#0 + 20*(j+1) #0
                yl = -10#30 + 10*(j+1) #-10
                l_special.append(j)
                if Label_cols[j] in ["C2", "C4"]:
                    xl = 0#-40 #0
                    yl = 5#-40  #5
                ax = OneAnnotation(ax, Label_cols[j], xy_cols[j, :], col_j, xl, yl, arrow = False)

            else:
                xl =0 #-50#0
                yl =5 #50#5
                
                if Label_cols[j] in ["C8", "C10"]:
                    xl = 0 #0
                    yl = 5  #5
                ax = OneAnnotation(ax, Label_cols[j], xy_cols[j, :], col_j, xl, yl, arrow = False)
                  
            
    return ax
            


def Display(Coords_rows, Coords_cols, Inertia, Data, rows_to_Annot, cols_to_Annot, Label_rows, Label_cols, 
            markers, col, figtitle, outliers, dtp, chosenAxes = np.array([0, 1]), show_inertia = True, reverse_axis = False, 
            separate = False, center = None, model = None, ColName = None, RowName = None, lims = True, plotRows = True, plotCols = True,
            fig = None, ax = None, give_ax = False, with_ref = None, cut_dist = {"shift_orig":(False, False), "cut_all":False}, log_log = False):  
    """
    @brief: display results
    """                           

    
    # plot 2 components
    PreFig()    
    if len(chosenAxes) == 2:
        dim1, dim2 = Inertia[chosenAxes]
    xy_rows = Coords_rows[:, chosenAxes] 
    xy_cols = Coords_cols[:, chosenAxes]
    
    if center is not None:
        if separate:
            center = (center[0][chosenAxes], center[1][chosenAxes])
        else:
            if len(chosenAxes) == 2:
                center = center[chosenAxes] 
            else:
                center = [center[chosenAxes]]
    
    not_orig = np.arange(0, xy_rows.shape[0]+xy_cols.shape[0]+1, 1, dtype = int) != xy_rows.shape[0]
    if with_ref is not None:
        try:
            with_ref = with_ref[:, chosenAxes]
            with_ref_c = with_ref
            
        except:
            with_ref_c = None   
    
    # cut them uniformly in each subspace without changing the angle between them (affine transformation preserving distance hierarchy)
    if cut_dist["cut_all"]:
        if len(chosenAxes) == 2:
            origin = center[np.newaxis, :]
            if with_ref is not None:
                coords = np.concatenate((xy_rows - origin, xy_cols - origin, with_ref[not_orig, :] - origin), axis = 0)
            else:
                coords = np.concatenate((xy_rows - origin, xy_cols - origin), axis = 0)
            norms = np.linalg.norm(coords, axis = 1)
            srt = np.argsort(norms)
            nu = 0.005
            cut = (1-nu)*norms[srt[0]]
            sph_coords, Transf_Mat = convert_spherical(coords, cut = cut)
            xy_rows = sph_coords[:xy_rows.shape[0], :] + origin
            xy_cols = sph_coords[xy_rows.shape[0]:xy_rows.shape[0]+xy_cols.shape[0], :] + origin
            if with_ref is not None:
                with_ref_c[not_orig, :] = sph_coords[xy_rows.shape[0]+xy_cols.shape[0]:,] + origin           
    
    # annotate points
    Rows_Labels = np.array([Label_rows[c] for c in Data.index], dtype = dtp[0])
    Cols_Labels = np.array([Label_cols[c] for c in Data.columns], dtype = dtp[1])
    
    if rows_to_Annot is not None:
        annot_rows = rows_to_Annot
        rows_to_Annot_index = []
        for s in range(len(annot_rows)):
            ind = np.where(Data.index == annot_rows[s])[0]
            if len(ind) >= 1: # should appear only one time
                rows_to_Annot_index = rows_to_Annot_index + list(ind)
    else:
        rows_to_Annot_index = None
                
    if cols_to_Annot is not None:
        annot_cols= cols_to_Annot
        cols_to_Annot_index = []
        for s in range(len(annot_cols)):
            ind = np.where(Data.columns == annot_cols[s])[0]
            if len(ind) >= 1: #  should appear only one time
                cols_to_Annot_index = cols_to_Annot_index + list(ind)
    else:
        cols_to_Annot_index = None
    
    if not separate:
        if fig == None or ax == None:
            fig = pl.figure(figsize=(36+18,20+10))#pl.figure(figsize=(18,10))
            ax = fig.add_subplot(2,1,1)
        
        if len(chosenAxes) == 1:
            xy_rows = np.concatenate((xy_rows, np.zeros(len(xy_rows))[:, np.newaxis]), axis = 1)
            xy_cols = np.concatenate((xy_cols, np.zeros(len(xy_cols))[:, np.newaxis]), axis = 1)
            rows_cols = np.concatenate((xy_rows[:, 0], np.array(center[0]), xy_cols[:, 0]))
            if with_ref is not None:
                with_ref_c = np.concatenate((rows_cols[:, np.newaxis], - np.abs(with_ref_c - center[0])), axis = 1) 
            
        if plotRows:
            try:
                # for coloring by cluster membership
                for j in range(xy_rows.shape[0]):
                    ax.scatter([xy_rows[j, 0]], [xy_rows[j, 1]], marker = markers[0][0], color = col[0][Rows_Labels[j]], s = markers[1][1])
            except:
                try:
                    col_cols = col[0]
                except:
                    print("Rows categories are green because you haven't entered an appropriate col parameter: it's either a tuple of 2 colors (one for all the row sand ones for all the cols) or a tuple of 2 dictionaries with labels as keys (one for all the rows and one for all the cols) ")
                    col_cols = "green"
                ax.scatter(xy_rows[:, 0], xy_rows[:, 1], marker = markers[0][0], color = col[0], s = markers[0][1], label= RowName)
            ax = Annotate(ax, rows_to_Annot_index, None, Rows_Labels, Cols_Labels, xy_rows, xy_cols, col, arrow = True)
            """
            if with_ref_c is not None:
                for rf in range(xy_rows.shape[0]):
                    pl.plot([xy_rows[rf, 0],with_ref_c[rf, 0]], [xy_rows[rf, 1], with_ref_c[rf, 1]], color = col[0], linewidth = 1)
            """
                
        
        if plotCols:
            try:
                # for coloring by cluster membership
                for j in range(xy_cols.shape[0]):
                    ax.scatter([xy_cols[j, 0]], [xy_cols[j, 1]], marker = markers[1][0], color = col[1][Cols_Labels[j]], s = markers[1][1])
            except:
                try:
                    col_cols = col[1]
                except:
                    print("Column categories are red because you haven't entered an appropriate col parameter: it's either a tuple of 2 colors (one for all the row sand ones for all the cols) or a tuple of 2 dictionaries with labels as keys (one for all the rows and one for all the cols) ")
                    col_cols = "red"
                
                ax.scatter(xy_cols[:, 0], xy_cols[:, 1], marker = markers[1][0], color = col_cols, s = markers[1][1], label= ColName)
            ax = Annotate(ax, None, cols_to_Annot_index, Rows_Labels, Cols_Labels, xy_rows, xy_cols, col, arrow = True)
            if with_ref is not None:
                for rc in range(xy_cols.shape[0]):
                    rf = xy_rows.shape[0] + 1 + rc
                    pl.plot([xy_cols[rc, 0],with_ref_c[rf, 0]], [xy_cols[rc, 1], with_ref_c[rf, 1]], color = col[1], linewidth = 1)
               
        
        if with_ref is not None:
            if plotRows and plotCols:
                ref_all = with_ref_c[not_orig, :]
                sort_all = np.argsort(ref_all[:, 0])
                ref_all[:, 0] = ref_all[sort_all, 0]
                ref_all[:, 1] = ref_all[sort_all, 1]
                pl.plot(ref_all[:, 0], ref_all[:, 1], color = "green", linewidth = 2)
            elif plotRows:
                ref_rows = with_ref_c[:xy_rows.shape[0], :]
                sortr = np.argsort(ref_rows[:, 0])
                ref_rows[:, 0] = ref_rows[sortr, 0]
                ref_rows[:, 1] = ref_rows[sortr, 1]
                pl.plot(ref_rows[:, 0], ref_rows[:, 1], color = col[0], linewidth = 2)
            elif plotCols:
                ref_cols = with_ref_c[xy_rows.shape[0]+1:, :]
                sortc = np.argsort(ref_cols[:, 0])
                ref_cols[:, 0] = ref_cols[sortc, 0]
                ref_cols[:, 1] = ref_cols[sortc, 1]
                pl.plot(ref_cols[:, 0], ref_cols[:, 1], color = col[1], linewidth = 2)
        
            if len(chosenAxes) == 1:
                not_orig = np.arange(0, with_ref_c.shape[0], 1, dtype = int) != xy_rows.shape[0]
                if plotRows and plotCols:
                    coords_1 = rows_cols[not_orig]
                elif plotRows:
                    coords = ref_rows
                elif plotCols:
                    coords_1 = ref_cols
                sort = np.argsort(coords_1)
                coords_1 = coords_1[sort]
                pl.plot(coords_1, -np.abs(coords_1 - center[0]), color = "orange", linewidth = 2)
            
        #ax.legend(loc= (1.05, 0))
        
         # label factor axis
        if show_inertia: # show percentage of inertia
            pl.xlabel("Dim %d (%.2f %%)"%(chosenAxes[0]+1, 100*dim1/np.sum(Inertia)), fontsize = 14)
            if len(chosenAxes) == 2:
                pl.ylabel("Dim %d (%.2f %%)"%(chosenAxes[1]+1, 100*dim2/np.sum(Inertia)), fontsize = 14)
            #pl.xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            #pl.ylabel("Dim %d"%(chosenAxes[1]+1,), fontsize = 14)
        else:
            pl.xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            if len(chosenAxes) == 2:
                pl.ylabel("Dim %d"%(chosenAxes[1]+1,), fontsize = 14)
            
        #ax = Separation_axis(ax, xy_rows, xy_cols, outliers, lims = True)
        if center is not None:
            #ax.plot([center[0]], [center[1]], "o", markersize = 10, color = "red", label = "I")
            
            if len(chosenAxes) == 2:
                if cut_dist["cut_all"]:
                    ### place a filled circle indicating that the distance to the new center is the same for all points
                    Circle = matplotlib.patches.Circle(xy = center, radius = (0.5*nu)*norms[srt[0]], facecolor = "green", alpha = 0.25)
                    ax.axvline(x = center[0], ls = "--", color = "black", linewidth =0.5)
                    ax.axhline(y = center[1], ls = "--", color = "black", linewidth =0.5)
                    ax.add_patch(copy(Circle))
                elif cut_dist["shift_orig"][0] or cut_dist["shift_orig"][1]:
                    origin = center[np.newaxis, :]
                    #center0 = center.copy()
                    coords = np.concatenate((xy_rows - origin, xy_cols - origin), axis = 0)
                    norms = np.linalg.norm(coords, axis = 1)
                    srt = np.argsort(norms)
                    nu = 0.0025
                    #nu = 0.0005
            
                    if cut_dist["shift_orig"][0]:
                        xmin, xmax = ax.get_xlim()
                        add_x = nu*np.sign(coords[srt[0], 0] - center[0])*np.abs(coords[srt[0], 0])
                        new_x = coords[srt[0], 0] - add_x
                        center[0] = new_x + center[0]
                        pl.plot([center[0], center[0] + add_x], [center[1], center[1]], ls = ":" , linewidth = 1, color = "green")# label= "c = %.2f"%np.abs(center0[0] - center[0]))
                        
                        x_0 = min(center[0], center[0]+add_x)
                        x_1 = max(center[0], center[0]+add_x)
                        pl.plot([xmin, x_0], [center[1], center[1]], ls = "--", color = "black", linewidth = 0.5, alpha = 0.5)
                        pl.plot([x_1, xmax+np.abs(add_x)], [center[1], center[1]], ls = "--", color = "black", linewidth = 0.5, alpha = 0.5)
                        if not cut_dist["shift_orig"][1]:
                            ax.axvline(x = center[0], ls = "--", color = "black", linewidth =0.5, alpha = 0.5)
                    else:
                        ymin, ymax = ax.get_ylim()
                        add_y =  nu*np.sign(coords[srt[0], 1] - center[1])*np.abs(coords[srt[0], 1])
                        new_y = coords[srt[0], 1] - add_y 
                        center[1] = new_y + center[1]
                        pl.plot([center[0], center[0]], [center[1], center[1] + add_y], ls = ":" ,linewidth = 1, color = "orange") #label = "c = %.2f"%np.abs(center0[1] - center[1]))
                        
                        y_0 = min(center[1], center[1]+add_y)
                        y_1 = max(center[1], center[1]+add_y)
                        pl.plot([center[0], center[0]], [ymin, y_0], ls = "--" ,linewidth = 0.5, color = "black", alpha = 0.5)
                        pl.plot([center[0], center[0]], [y_1, ymax+np.abs(add_y)], ls = "--" ,linewidth = 0.5, color = "black", alpha = 0.5)
                        if not cut_dist["shift_orig"][1]:
                            ax.axhline(y = center[1], ls = "--", color = "black", linewidth =0.5, alpha = 0.5)
                    
                else:  
                    ax.axvline(x = center[0], ls = "--", color = "black", linewidth =0.5)
                    ax.axhline(y = center[1], ls = "--", color = "black", linewidth =0.5)
               
            if lims:
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                
                # remove extreme outliers from the figure
                """
                if len(chosenAxes) == 2:
                    dist_xn = scp.spatial.distance.pdist(xy_rows)
                    dist_yn = scp.spatial.distance.pdist(xy_cols)
                    dist = np.concatenate((dist_xn, dist_yn))
                    iqr_dist = np.percentile(dist, 20, interpolation = "midpoint")
                    xmin, xmax = -iqr_dist, iqr_dist
                    ymin, ymax = -iqr_dist, iqr_dist
                elif len(chosenAxes) == 1:
                    xmin, xmax = -max(np.abs(rows_cols[rows_cols<0]))/2, max(np.abs(rows_cols[rows_cols<0]))/2
                    ymin, ymax = ymin, 1.5*ymax
                """
                pl.xlim((xmin, xmax))
                pl.ylim((ymin, ymax))
            
        else:
            ax = Separation_axis(ax, xy_rows, xy_cols, outliers, lims = lims, col = "grey")
            
        # aspect ratio of axis
        #ax.set_aspect(1.0/(1.25*ax.get_data_ratio()), adjustable='box') # this can deforms the entire configuration, distances that are equal may look different and vice-versa
        
        if log_log:
            ax.set_yscale("symlog")
            ax.set_xscale("symlog")
        else:
            ax.set_aspect("equal")
        ax.set_xticks(())
        ax.set_yticks(())
        
        """
        if model is not None:
            if model["model"] == "x|y":
                ax.set_title("How present is %s ($X$) within variable %s ($Y$)? ($X|.$) \n How similar is the occurence of %s ($X$) among  %s ($Y$)? ($.|Y$) \n The distance between variables %s and %s \n shows the strengh of their relationships"%(ColName, RowName, ColName, RowName, ColName, RowName))
            elif model["model"] == "y|x":
                ax.set_title("How present is %s ($Y$) within variable %s ($X$) ? ($Y|.$) \n How similar is the occurence of %s ($Y$) among %s ($X$)? ($.|X$) \n The distance between variables %s and %s \n shows the strengh of their relationships"%(RowName, ColName, RowName, ColName, ColName, RowName))
            elif model["model"] == "ca_stand":
                ax.set_title("How similar is the occurence of %s ($Y$) among %s ($X$)? ($.|X$) \n How similar is the occurence of %s ($X$) among %s ($Y$)? ($.|Y$) \n The distance between variables %s and %s has no meaning?"%(RowName, ColName, ColName, RowName, ColName, RowName))
            elif model["model"] == "presence":
                ax.set_title("How present is %s ($Y$) within variable %s ($X$) ? ($Y|.$) \n How present is %s ($X$) within variables %s ($Y$)? ($X|.$) \n The distance between variables %s and %s \n shows the strengh of their relationships"%(RowName, ColName, ColName, RowName, ColName, RowName))
            elif model["model"] == "stand":
                ax.set_title("How similar is the occurence of %s ($Y$) among %s ($X$)? ($.|X$)"%(RowName, ColName)+"\n How similar is the occurence of %s ($X$) among  %s ($Y$)? ($.|Y$)"%(ColName, RowName)+ "\n The distance between variables %s and %s \n shows the strengh of their relationships"%(RowName, ColName))
                
            pl.suptitle(figtitle)
        else:
            pl.title(figtitle)
        """
        
        gs = None # just a placeholder
        
        ax.axis("off")
        
        pl.suptitle(figtitle)
        
    else:  # Separated
        fig = pl.figure(constrained_layout=True, figsize = (7, 9))
        gs = fig.add_gridspec(2, 2)
        pl.suptitle(figtitle)
        
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 0])
    
        ax1.scatter(xy_rows[:, 0], xy_rows[:, 1], marker = markers[0][0], color = col[0], s = markers[0][1])
        ax2.scatter(xy_cols[:, 0], xy_cols[:, 1], marker = markers[1][0], color = col[1], s = markers[1][1])
        
        xmin, xmax = 1.5*np.amin(xy_rows[:, 0]), 1.5*np.amax(xy_rows[:, 0])#ax1.get_xlim()
        ymin, ymax = 1.5*np.amin(xy_rows[:, 1]), 1.5*np.amax(xy_rows[:, 1]) #ax1.get_ylim()
        
        xmin2, xmax2 = 1.5*np.amin(xy_cols[:, 0]), 1.5*np.amax(xy_cols[:, 0]) #ax2.get_xlim()
        ymin2, ymax2 = 1.5*np.amin(xy_cols[:, 1]), 1.5*np.amax(xy_cols[:, 1])#ax2.get_ylim()
        
        try:
            ax1 = Annotate(ax1, rows_to_Annot_index, None, Rows_Labels, Cols_Labels, xy_rows, xy_cols, col, arrow = True)
            ax2 = Annotate(ax2, None, cols_to_Annot_index,  Rows_Labels, Cols_Labels, xy_rows, xy_cols, col, arrow = True)
        except:
            print("No annotations")

        #ax1.legend(loc= "best", fontsize = 10)
        #ax2.legend(loc= "best", fontsize = 10)
        if model is not None:
            if model["model"] == "x|y":
                ax1.set_title("Similarity within Y? ($.|Y$)") #How important is X for variable Y?
                ax2.set_title("$X|.$") #
            elif model["model"] == "y|x":
                ax1.set_title("$Y|.$")  #How important is Y for variable X?
                ax2.set_title("Similarity within X? ($.|X$)") 
            elif model["model"] == "presence":
                ax1.set_title("$Y|.$")  #How important is Y for variable X?
                ax2.set_title("$X|.$") # How similar is the occurence of Y within X? 
            else:
                 #ax1.set_title("Similarity within Y? ($.|Y$)") #How similar is the occurence of X within Y? 
                 #ax2.set_title("Similarity within X? ($.|X$)") #How similar is the occurence of Y within X?
                ax1.set_title("$.|Y$")  #How important is Y for variable X?
                ax2.set_title("$.|X$") 
        else:
            ax1.set_title(RowName)
            ax2.set_title(ColName)
        
        # label factor axis
        if show_inertia: # show percentage of inertia
            #pl.xlabel("Dim %d (%.2f %%)"%(chosenAxes[0]+1, 100*dim1/np.sum(Inertia)), fontsize = 14)
            #pl.ylabel("Dim %d (%.2f %%)"%(chosenAxes[1]+1, 100*dim2/np.sum(Inertia)), fontsize = 14)
            ax1.set_xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            ax2.set_xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            ax2.set_ylabel("Dim %d"%(chosenAxes[1]+1,), fontsize = 14)
        else:
            ax1.set_xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            ax2.set_xlabel("Dim %d"%(chosenAxes[0]+1,), fontsize = 14)
            ax2.set_ylabel("Dim %d"%(chosenAxes[1]+1,), fontsize = 14)
            
        
        
        if center is not None: 
             ax1.axvline(x = center[0][0], ls = "--", color = "black", linewidth =0.5)
             ax1.axhline(y = center[0][1], ls = "--", color = "black", linewidth =0.5)
        
             ax2.axvline(x = center[1][0], ls = "--", color = "black", linewidth =0.5)
             ax2.axhline(y = center[1][1], ls = "--", color = "black", linewidth =0.5)
        else:
           ax1.axvline(x = 0, ls = "--", color = "black", linewidth =0.5)
           ax1.axhline(y = 0, ls = "--", color = "black", linewidth =0.5)
           
           ax2.axvline(x = 0, ls = "--", color = "black", linewidth =0.5)
           ax2.axhline(y = 0, ls = "--", color = "black", linewidth =0.5)
       
        # aspect ratio of axis
        #ax1.set_aspect(1.0/(1.25*ax1.get_data_ratio()), adjustable='box')
        #ax2.set_aspect(1.0/(1.25*ax2.get_data_ratio()), adjustable='box')
        
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        
        ax1.set_xlim((xmin, xmax))
        ax1.set_ylim((ymin, ymax))
        
    
        ax2.set_xlim((xmin2, xmax2))
        ax2.set_ylim((ymin2, ymax2))
        
        ax1.set_xticks(())
        ax1.set_yticks(())
        
        ax2.set_xticks(())
        ax2.set_yticks(())
        
        ax1.axis("off")
        ax2.axis("off")
        pl.suptitle(figtitle)
      
    if give_ax == True:
        return fig, ax, xy_rows, xy_cols, gs, center
    else:         
        return fig, xy_rows, xy_cols, gs, center


def seriation_curves(Curves, Coords, center, Data, sort, labels, which_var = "columns"):
    PreFig()
    fig = pl.figure(figsize = (7, 7))
    cKeys = list(Curves.keys())
    Coords = Coords[sort, :]
    
    if which_var == "columns":
        ylabels = Data.columns[sort]
    else:
        xlabels = Data.index[sort]
    
    num = 1
    for k in range(len(cKeys)):
        if cKeys[k]!="dim1_var" and Curves[cKeys[k]][1]<=4:
            ax = fig.add_subplot(2,2,num)
            y_k = Curves[cKeys[k]][0]
            iyk = Curves[cKeys[k]][1]

            y = Coords[:, iyk]
            
            ax.scatter(Coords[:, 0], y, s = 10, marker = "o", color = "red")
            
            pl.plot(Curves["dim1_var"], y_k, linewidth = 1, color = "green")
            
            if which_var == "columns":
                for i in range(len(Data.columns)):
                    ax = OneAnnotation(ax, labels[ylabels[i]], (Coords[i, 0], Coords[i, iyk]), col_val = "white", xl=-5, yl=-15, arrow = True, fontsize = 4, alpha = 0.5)
            else:
                for i in range(len(Data.index)):
                    ax = OneAnnotation(ax, labels[xlabels[i]], (Coords[i, 0], Coords[i, iyk]), col_val = "white", xl=-5, yl=-15, arrow = True, fontsize = 0.5)
            
            ax.axvline(x = center[0], ls = "--", color = "black", linewidth = 0.5)
            ax.axhline(y = center[1], ls = "--", color = "black", linewidth = 0.5) 
            pl.title(cKeys[k])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            num += 1
            
    return fig