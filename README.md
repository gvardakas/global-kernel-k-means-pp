# Efficient error minimization in kernel k-means clustering

Kernel ***k***-means extends the ***k***-means algorithm to identify non-linearly separable clusters but is inherently sensitive to cluster initialization. To address
this challenge, we first formulate the kernel ***k***-means++ method, which conveys
the efficient center initialization strategy of ***k***-means++ from Euclidean to kernel space. Building on this, we propose global kernel ***k***-means++ (GK***k***M++), a
novel clustering algorithm designed to balance clustering error minimization with
reduced computational cost. GK***k***M++ extends the well-established global kernel
***k***-means algorithm by incorporating the stochastic initialization strategy of kernel
***k***-means++. This approach significantly reduces computational complexity while
preserving superior clustering error minimization capabilities akin to traditional
global kernel ***k***-means. The experimental results on synthetic, real, and graph
datasets indicate that GK***k***M++ consistently outperforms both kernel ***k***-means
with random initialization and kernel ***k***-means++, while achieving solutions comparable to those provided by the exhaustive and computational intensive global
kernel ***k***-means method.


# Execution Instructions
1) You should first install all requirements via running in your terminal the command pip install -r requirements.txt.
2) You should open the Kernel k-means.ipynb, Kernel k-means_Graphs.ipynb and Kernel k-means_Rings.ipynb, and for each one of them run each cell of it via a suitable program e.g. Visual Code.  
