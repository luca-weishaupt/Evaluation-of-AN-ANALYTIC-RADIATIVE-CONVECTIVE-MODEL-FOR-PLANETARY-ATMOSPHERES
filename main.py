from functions import *

if not os.path.exists("./figures"):
    os.makedirs("./figures")

if __name__ == '__main__':
    # Plot Figure 1
    fig1(dest="./figures/fig1.png")
    fig2(dest="./figures/fig2.png")
    fig3(dest="./figures/fig3.png")
    fig8(dest="./figures/fig8.png")
    figEarth(dest="./figures/earth.png")