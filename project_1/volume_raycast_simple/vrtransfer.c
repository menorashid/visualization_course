#include "vrtransfer.h"


//creates a very simple transfer function that linearly
//interpolates color and alpha values between min and
//max such that min -> RGBA (0,0,0,0) and max -> RGBA(1,1,1,1);
int
vrt_init(TFUNC* tf, int length, float min, float max,uint8_t * colormap){
    if(max <= min || tf==NULL){
        return 0;
    }

    tf->colors = (glm::vec4*)malloc(length*sizeof(glm::vec4));
    if(!tf->colors){
        return 0;
    }

    tf->min_val = 0.;
    tf->max_val = 1.;
    tf->min_thresh = 0.;
    tf->max_thresh = 1.;

    tf->max = max;
    tf->min = min;
    tf->len = length;
    EPRINT("len %d",tf->len);

    float inc = (max-min)/(length-1);
    for(int i =0; i < length; i++){
        //grey scale transfer function with linearly
        //ramping alpha values;
        float val = inc*i;

        int ind_curr = 4*i;
        float r = ((float) colormap[ind_curr])/ 255.;
        float g = ((float) colormap[ind_curr+1])/ 255.;
        float b = ((float) colormap[ind_curr+2])/ 255.;
        float a = ((float) colormap[ind_curr+3])/ 255.;
        // printf ("%d,%d,%d,%d\n",colormap[ind_curr],colormap[ind_curr+1],colormap[ind_curr+2],colormap[ind_curr+3]);
        // printf ("%f,%f,%f,%f\n",r,g,b,a);

        tf->colors[i] = glm::vec4(r,g,b,val);
        // tf->colors[i] = glm::vec4(1.-val,0.5*val,val,val);
    }
    return 1;
}

uint8_t* readcolormap(const char *fn,int colormap_size)
{
    FILE *fd;
    fd = fopen(fn,"rb");
    if(!fd){
        EPRINT("Error opening file %s\n",fn);
    }

    uint8_t *colormap = (uint8_t *)malloc(colormap_size*4);
    // float *gx_in = (float *)malloc(gridx*gridy*gridz*3*sizeof(float));

    // float *tri_in = (float *)malloc(gridx*gridy*gridz*4*4*4*sizeof(float));
    
    if(fread(colormap,1,colormap_size*4,fd) < colormap_size*4){
        free(colormap);
        EPRINT("Error reading in volume %s\n",fn);
    }
    fclose(fd);
    return colormap;
}

void
vrt_clean(TFUNC* tf){
    free(tf->colors);
    tf->colors = NULL;
    tf->len = 0;
    tf->min = MAXFLOAT;
    tf->max = -MAXFLOAT;
}

/****************************************************************************************************
 * The code was developed by Garrett Aldrich for [ECS 277] Advanced Visualization at UC Davis.
 * Bugs and problems :'(
 * If you are in my class, please don't email me.... start a thread on canvas :-)
 * If you aren't in my class I apologize for the bad code, I probably will never fix it!
 *
 * It's free as in beer
 * (free free, do whatever you want with it)
 *
 * If you use a big chunk, please keep this code block somewhere in your code (the end is fine)
 * Or atleats a comment my name and where you got the code.
 *
 * If you found this useful please don't email me, (sorry I ignore way to many already),
 * feel free to send me a thanks on instagram or something like that (I might not read it for a
 * while, but eventually I will)
 *
 ****************************************************************************************************/
