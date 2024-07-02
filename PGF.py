# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:48:54 2023

@author: paulg
"""

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import subprocess
import os
import pandas as pd
# import seaborn as sns
import scipy


def MeanStd_FromGroups(df, column_name): 
    """
    takes a dataframe (df) an returns the mean and std values of the groups given by the common values of column_name (string) 
    returns the array of unique values in the column of column_name, and thw dataframes containing the mean and std of all the columns
    """
    df_mean = df.groupby(column_name).mean()
    df_std = df.groupby(column_name).std()

    x = np.asarray([np.asarray(i) for i in df_mean.index.values]).squeeze()
    return x, df_mean, df_std


def ColorMix(cmap1='Oranges', cmap2 = 'Greens', weight1=.7, level1=.5, level2=None):
    '''
    cmap1 and cmap2 should be strings of cmaps (eg 'viridis' or 'plasma' or 'Greens')
    fact is the factor between cmap1 and cmap2 (1 is full cmap1, 0 is full cmap2)
    intensity is how high we go up in orange and viridis (0 is probably white, 1 is the end of the colormap)
    outputs: a color (array with 3 doubles) and alpha=1
    '''   
    
    if level2 == None:
        level2 = level1
    
    colormap1 = cm.get_cmap(cmap1)
    colormap2 = cm.get_cmap(cmap2)
    
    color1 = np.array(colormap1(level1))
    color2 = np.array(colormap2(level2))

    return weight1 * color1 + (1-weight1) * color2


def SaveFig(savefigname='Figure', dpi=480, transparent=True, type_list=['png','svg', 'pdf'], dir=None):
    """
    Saves a figure as an svg and a png file
    savefigname: default 'Figure', string, no extension
    dpi: default 480, dots per inches for the png export
    dir: directory where you want to save the images
    """
#    plt.tight_layout()

    current_directory = os.getcwd()
    if dir!=None:
        os.chdir(dir)
    savefigname = savefigname
    for pngsvg in type_list:
        savestring = './' + savefigname+'.'+pngsvg
        plt.savefig(savestring,dpi=dpi,format=pngsvg, bbox_inches='tight',  pad_inches=.1, transparent=transparent)
        #plt.savefig(savestring,dpi=dpi,format=pngsvg,  transparent=True)
    if dir!=None:
        os.chdir(current_directory)
    #     # files.download(savestring) 
    
def OpenExplorer(filepath=None): 
    if filepath==None:
        filepath = os.getcwd()

    subprocess.Popen('explorer ' + filepath)

    
def Convert_xlsx2csv(filename, delete_xlsx=False): 
    
    if filename[-5:] == '.xlsx':
        filename = filename[:-5]

    filename_without_extension = filename
    
    T = pd.read_excel(filename_without_extension + '.xlsx')
    T.to_csv(filename_without_extension + '.csv')
    print('Success converting ' + filename_without_extension + '.csv\n')
    if delete_xlsx:
        os.remove(filename_without_extension + '.xlsx')
        
    
def SetDefaultTicksIn(ax=None):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    
    
def SetAxSize(w=4,h=3, ax=None, unit='inches', user='None'):   
    """ 
    SetAxSize(w=4,h=3, ax=None, unit='inches', user='None'): w, h: width, height in inches
    """   
    if not ax: ax=plt.gca()
    
        
    if user == 'Paul': 
        print('SetAxSize: user=Paul - any other attribute will be overwritten')
        w = 5
        h = w*3/4
        unit = 'cm'
        
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    #figw = float(w)+ r + l
    figh = float(h)/(t-b)
    #figh = float(h) + t + b
    
    
    if unit=='inches':
        ax.figure.set_size_inches(figw, figh)
        
    if unit=='mm':
        ax.figure.set_size_inches(figw/25.4, figh/25.4)
        
    if unit=='cm':
        ax.figure.set_size_inches(figw/2.54, figh/2.54)




# SetAxeSize2


def SetAxeSize2(w=4,h=3, ax=None, fig=none, unit='inches', user='None'): 
    """ 
    SetAxeSize2(w=4,h=3, ax=None, fig=none, unit='inches', user='None'): w, h: width, height in inches
    """   

    if not ax: ax=plt.gca()

    if not fig: fig=plt.gcf()
        
        
            
    if user == 'Paul': 
        print('SetAxSize: user=Paul - any other attribute will be overwritten')
        w = 5
        h = w*3/4
        unit = 'cm'
        

        fig_bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig_width, fig_height = fig_bbox.width, fig_bbox.height

        ax_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax_width, ax_height = ax_bbox.width, ax_bbox.height    
        
        figw = fig_width * target_ax_width / ax_width
        figh = fig_height * target_ax_height / ax_height

        if unit=='inches':
            ax.figure.set_size_inches(figw, figh)
            
        if unit=='mm':
            ax.figure.set_size_inches(figw/25.4, figh/25.4)
            
        if unit=='cm':
            ax.figure.set_size_inches(figw/2.54, figh/2.54)
        
        
        
def SetDefaultFont(font='Arial', stretch = 'condensed', editable_fonts=True, user=None):
    """
    Sets the default font for figures that use matplotlib
    font: string (eg, 'Myriad Pro')
    stretch: 'normal' or 'condensed'
    editable_fonts: boolean, allows you to edit text in illustrator or inkscape
    """
    
    plt.rcParams['font.family'] = 'sans-serif'
    
    plt.rcParams['font.sans-serif'] = font
    plt.rcParams['font.stretch'] = stretch
    
    if editable_fonts:
        matplotlib.rcParams['pdf.fonttype'] = 42 ; # pdf.fonttype : 42 # Output Type 3 (Type3) or Type 42 (TrueType)
        matplotlib.rcParams['ps.fonttype'] = 42
    
    if user=='paul':
        plt.rcParams['font.sans-serif'] = 'Myriad Pro'
        plt.rcParams['font.stretch'] = 'condensed'
        
        params = {'legend.fontsize': 8,
         'axes.labelsize':9,
         'axes.titlesize':9,
         'xtick.labelsize':8,
         'ytick.labelsize':8}
        plt.rcParams.update(params)


"""# stats"""
def my_stats(nested_array):
    '''
    stats takes a nested array to generate a dataframe with the means and standard deviations of each sub-array contained in the nested array
    args:     nested_array
              array of arrays that do not have the same length, e.g., np.array([[1,2,3],[1,2,3,4]], dtype='object')
     [[1,2,3],[1,2,3,4]] as input and outputs a panda dataframe with .means and .stds of each array of the nested array

    returns:  df
              dataframe where df.means and df.stds r|eturn the means and stds of the each array of the nested array

    Example:
        x = stats(np.array([[1,2,3],[1,2,3,4]], dtype='object'))
        print(x.means)
        print(x.stds)

    returns
        0    2.0
        1    2.5
        Name: means, dtype: float64
        0    0.816497
        1    1.118034
        Name: stds, dtype: float64
    '''
    means = np.empty(np.shape(nested_array)[0])
    stds = np.empty(np.shape(nested_array)[0])

    for i in range(np.shape(nested_array)[0]):

        if np.isnan(np.sum(nested_array[i])):
            means[i] = np.nanmean(nested_array[i])
            stds[i] = np.nanstd(nested_array[i])
        else:
            means[i] = np.mean(nested_array[i])
            stds[i] = np.std(nested_array[i])

    df = pd.DataFrame({
        'means': means,
        'stds': stds
    })

    return df

"""# my_boxplot"""
# Update by Tim (02-28-2022): Added alpha and color options. Use list of RGB color codes that are the same lengths as your list of experiments.

def my_boxplot(list_of_arrays, exp_names, alpha = 0.2,
               my_color_list = ['r'], fact_for_y_shift = 0.05,
               markersize=40, ax=None, text_font_size=10, annotate=True,
               format_mean = '%.3g', format_std='%.2g', widths=None):
    '''
    list_of_arrays should be a list looking like this: [np.array([1,2,4,5]),np.array([2,2,5])]
    Modified by Paul 2023-05-11: You can now provide an ax to do the plotting (this function does not
                                                                                   generate its own figure anymore)
    my_boxplot outputs a figure with the boxplot of the nested_array of a nested array (array of arrays).
    exp_names is a list of strings for the xticks
    '''
    #plt.figure(figsize=figsize)
    if ax==None:
        ax = plt.gca()
        
    ax.boxplot(list_of_arrays,
              #patch_artist = True,
              #boxprops=dict(facecolor=my_color, color=my_color),
              medianprops=dict(color='k'),
              widths = widths,
              )
    ax.set_xticks(range(1,len(exp_names)+1), exp_names, fontsize=12)

    for i in range(0,len(exp_names)):
        y = list_of_arrays[i]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(i+1, 0.01, size=len(y))
        
#        if isinstance(my_colors, str):
        if hasattr(my_color_list, '__len__'):
            if len(my_color_list) == 1:
                ax.scatter(x, y, markersize, color=np.array([my_color_list[0]]),  alpha = alpha, edgecolors = 'none')
            else:   
                ax.scatter(x, y, markersize, color=np.array([my_color_list[i]]),   alpha = alpha, edgecolors = 'none')
        else: 
            ax.scatter(x, y,  markersize, color=np.array([my_color_list]), alpha = alpha, edgecolors = 'none')
            
            
#        else:
#            ax.plot(x, y, '.', alpha = my_alpha, color = my_colors[i],
#                    markersize=my_markersize, markeredgewidth=0.0)


    nested_array = np.array(list_of_arrays, dtype = 'object') # transform list of np.arrays into an np.array of np.arrays

    mean_vec = my_stats(nested_array).means
    std_vec = my_stats(nested_array).stds

    the_maxes = np.zeros(np.shape(mean_vec))
    #y_shift = np.max(mean_vec)*.2
    for i, v in enumerate(mean_vec):
        #y_shift = np.max(mean_vec)*.2
        y_shift = np.max(nested_array[0])*fact_for_y_shift
        the_maxes[i]=  np.max(nested_array[i])
        if annotate:
            ax.text(i+1 , np.max(nested_array[i]) + y_shift, format_mean% mean_vec[i] + '\n$\pm$' + format_std%std_vec[i],
                    color='black',  ha='center', fontsize=text_font_size)

    ax.set_ylim([0, np.max(the_maxes)*1.2+ y_shift])


    ax.set_ylim(0, None)
    
    plt.gcf().autofmt_xdate()

    # Group boxplot (Tim)
"""# my_boxplot_grp"""

def my_boxplot_grp(nested_array, group_names, category_names, my_alpha = 0.2, category_colors = 'r', marksize = 4):

    N_g           = len(group_names)
    grp_ticks     = np.zeros(N_g)



    for i in range(0,N_g):
        plt.boxplot(nested_array[2*i],
                    positions = [(3*i + 1)],
                    widths = 0.6,
                    medianprops=dict(color='k')
                    )
        plt.boxplot(nested_array[2*i+1],
                    positions = [(3*i + 2)],
                    widths = 0.6,
                    medianprops=dict(color='k')
                    )
        grp_ticks[i]  = 3*i + 1.5

        plt.xticks(grp_ticks, group_names, fontsize=12)

    for i in range(0,N_g):
        y1    = nested_array[2*i]
        y2    = nested_array[2*i + 1]
        # Add some random "jitter" to the x-axis
        x1    = np.random.normal((3*i + 1), 0.04, size=len(y1))
        x2    = np.random.normal((3*i + 2), 0.04, size=len(y2))
        if category_colors == 'r':
            plt.plot(x1, y1, '.', alpha = my_alpha, markersize = marksize)
            plt.plot(x2, y2, '.', alpha = my_alpha, markersize = marksize)
        if i is N_g-1:
            plt.plot(x1, y1, '.', alpha = my_alpha, label = category_names[0], markersize = marksize)
            plt.plot(x2, y2, '.', alpha = my_alpha, label = category_names[1], markersize = marksize)
        else:
            plt.plot(x1, y1, '.', alpha = my_alpha, color = category_colors[0], markersize = marksize)
            plt.plot(x2, y2, '.', alpha = my_alpha, color = category_colors[1], markersize = marksize)
        if i is N_g-1:
            plt.plot(x1, y1, '.', alpha = my_alpha, color = category_colors[0], label = category_names[0], markersize = marksize)
            plt.plot(x2, y2, '.', alpha = my_alpha, color = category_colors[1], label = category_names[1], markersize = marksize)



    mean_vec  = my_stats(nested_array).means
    std_vec   = my_stats(nested_array).stds

    the_maxes = np.zeros(np.shape(mean_vec))
    #y_shift = np.max(mean_vec)*.2
    for i in range(0,N_g):
        y_shift           = np.max(nested_array[0])*.05
        the_maxes[2*i]    = np.max(nested_array[2*i])
        the_maxes[2*i+1]  = np.max(nested_array[(2*i)+1])
        plt.text((3*i + 1) , np.max(nested_array[2*i]) + y_shift, '%.2f'% mean_vec[2*i] + '\n$\pm$' + '%.2f' %std_vec[2*i], color='black',  ha='center')
        plt.text((3*i + 2) , np.max(nested_array[2*i+1]) + y_shift, '%.2f'% mean_vec[2*i+1] + '\n$\pm$' + '%.2f' %std_vec[2*i+1], color='black',  ha='center')

    plt.ylim([0, np.max(the_maxes)*1.2+ y_shift])
    plt.xlim([-0.5, N_g*3 + 0.5])

    leg   = plt.legend(loc=8, prop={'size': 14}, ncol = 2, frameon=False)
    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(1)
        lh._legmarker.set_markersize(14)

        plt.ylim(0, None)
        plt.gcf().autofmt_xdate()

"""# read_table_NI_simple"""

def read_table_NI_simple(exp_names):
    '''
    read_table_NI_simple takes the exp_names (list of strings without '.csv') of the experiments and generates nested arrays for the reduced modulus, E, in MPa, and the hardness, H, in MPa.
    The files exp_names.csv are obtained by exporting the results in the femtotools software into a csv file.
    '''
    E = []
    H = []

    for k,exp_name in enumerate(exp_names):
        print(exp_name)
        table = pd.read_csv(exp_name + '.csv')
        x = table['Comment']
        E.append(np.array(table[['Reduced Modulus [MPa]']][pd.isna(x)]))
        H.append(np.array(table[['Hardness [MPa]']][pd.isna(x)]))

    E = np.array(E, dtype=object)/1000#making this GPA
    H = np.array(H, dtype=object)

    return E, H




# def plot_t_test_matrix(set_of_values, DOEs, my_color, annot=True):
#     N = np.shape(DOEs)[0]

#     t_statistic = np.zeros([N, N])
#     t_pval = np.zeros_like(t_statistic)

#     for i in  range(N):
#         for j in range(N):
#             t_statistic[i,j], t_pval[i,j] = scipy.stats.ttest_ind(set_of_values[i], set_of_values[j], equal_var = False)
#             my_color_rgb = np.array(matplotlib.colors.to_rgb(my_color))

#     boundaries = [0, 0.001, 0.01, 0.05, 1]
#     colors = [1 - (1-my_color_rgb),
#             1 - (1-my_color_rgb)/1.5 ,
#             1 - (1-my_color_rgb)/3,
#             1 - (1-my_color_rgb)/20,
#             #'w'
#             ]

#     norm = matplotlib.colors.BoundaryNorm(boundaries=boundaries, ncolors=256)

#     cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

#     # Getting the Upper Triangle of the co-relation matrix
#     matrix = np.triu(t_pval)
#     print(matrix)
#     # using the upper triangle matrix as mask
#     #sns.heatmap(corr, annot=True, mask=matrix)

#     sns.heatmap(t_pval,
#               linewidths=0.2,
#               linecolor='k',
#               annot=annot,
#               annot_kws={"size": 9, "color":'k'},
#               cmap=cmap,
#               fmt=".2e",
#               norm=norm,
#               xticklabels=DOEs,
#               yticklabels=DOEs,
#               #mask=matrix,
#               cbar_kws={'label': 'p value'})

#     plt.show()
#     return t_pval

### Functions for Aesthetics

class SpecialPrint:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def PrintColor(text='test'):
    """
    using
    class print_color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    """
    
    print(SpecialPrint.RED + SpecialPrint.BOLD + 
          text + SpecialPrint.END + SpecialPrint.END)
    
    
    
def replace_file_extension(file_path, new_extension):
    """
    Replace the file extension in a file path with a new extension.
    
    Args:
    - file_path (str): The path of the file.
    - new_extension (str): The new extension to replace the existing one.
    
    Returns:
    - str: The modified file path with the new extension.
    """
    # Split the file path and extension
    file_name, old_extension = os.path.splitext(file_path)
    
    # Combine the file name with the new extension
    new_file_path = f"{file_name}{new_extension}"
    
    return new_file_path



