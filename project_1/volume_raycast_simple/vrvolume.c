#include "vrvolume.h"

#define MAX_UI8 255.0

float nCr[4] = {1.,3.,3.,1.};
float nCr_2[3] = {1.,2.,1.};
float epsilon = 0.0000001; // to prevent zero mag gradients
/*
 *  Returns the value of the nearest grid point to "pt"
 */
float
interpolate_nearest_ui8(const VRVOL* vol, glm::vec3 pt)
{
    if(pt.x < 0 || pt.y < 0 || pt.z < 0 ||
       pt.x >= vol->gridx || pt.y >= vol->gridy || pt.z >= vol->gridz)
    {
        return 0.0; //this treats everything outside the volume as 0
                    //there are other choices, but this will work for
                    //most cases.
    }

    int ind = (int)roundf(pt.x);
    ind += (int)roundf(pt.y)*vol->gridx;
    ind += (int)roundf(pt.z)*vol->gridx*vol->gridy;
    float val =  ((uint8_t*)vol->data)[ind]/MAX_UI8;
    return val;
}


float get_val_at_index(const VRVOL* vol,int x, int y, int z){
// (const VRVOL* vol,float x, float y, float z){
    // int ind = (int)roundf(x);
    // ind += (int)roundf(y)*vol->gridx;
    // ind += (int)roundf(z)*vol->gridx*vol->gridy;

    int ind = x;
    ind += y*vol->gridx;
    ind += z*vol->gridx*vol->gridy;
    float val =  ((uint8_t*)vol->data)[ind];
    return val;
}

glm::vec3
get_grads_at_index_works(const VRVOL* vol,int x, int y, int z){

    int idx_less;
    int idx_more;
    float grad_curr;
    
    int idx_list[] = {x,y,z};
    size_t size_list[] = {vol->gridx,vol->gridy,vol->gridz};
    glm::vec3 grad_total = glm::vec3(0.,0.,0.);

    for (int curr_dim=0; curr_dim<3; curr_dim++){

        idx_less = idx_list[curr_dim] - 1;
        idx_more = idx_list[curr_dim] + 1;

        if (idx_less<0){
            idx_less = idx_list[curr_dim];
        }

        if (idx_more>=size_list[curr_dim]){
            idx_more = idx_list[curr_dim];
        }
        
        // (idx_less < 0) ? (idx_less = idx_list[curr_dim]);
        // (idx_more > size_list[curr_dim]) ? (idx_more = idx_list[curr_dim]);

        if (curr_dim==0){
            grad_curr = (get_val_at_index(vol, idx_less, y, z)+ get_val_at_index(vol, idx_more, y, z))/2.;
            grad_total.x = grad_curr;
        }
        else if (curr_dim==1){
            grad_curr = (get_val_at_index(vol, x, idx_less, z)+ get_val_at_index(vol, x, idx_more, z))/2.;
            grad_total.y = grad_curr;
        }
        else{
            grad_curr = (get_val_at_index(vol, x, y, idx_less)+ get_val_at_index(vol, x, y, idx_more))/2.;
            grad_total.z = grad_curr;    
        }
    }

    return grad_total;

}


void 
calculate_coeff_at_index(const VRVOL* vol,int x0, int y0, int z0){

    int x1 = x0+1;
    int y1 = y0+1;
    int z1 = z0+1;
    float coeff[4][4][4];
    int ind;
    float coeff_curr;

    int coord[3][2]={  {x0,x1},{y0,y1},{z0,z1}};
    // int coord[3][2]={  {0,1},{0,1},{0,1}};

    //make cubes
    float corner_data[2][2][2][4];
    float f_curr;
    glm::vec3 grad_curr;

    if (x1>=vol->gridx || y1>=vol->gridy || z1>=vol->gridz ){
        for (int i = 0; i<4; i++){
            for (int j =0;j<4; j++){
                for (int k = 0;k<4; k++){
                    coeff[i][j][k]=0.;
                }
            }
        }
    }
    else{
        
        for (int x_ind=0;x_ind<2;x_ind++){
            for (int y_ind=0;y_ind<2;y_ind++){
                for (int z_ind=0;z_ind<2;z_ind++){

                    f_curr = get_val_at_index(vol,coord[0][x_ind],coord[1][y_ind],coord[2][z_ind]);
                    grad_curr = get_grads_at_index(vol,coord[0][x_ind],coord[1][y_ind],coord[2][z_ind]);
                    corner_data[x_ind][y_ind][z_ind][0] = f_curr;
                    corner_data[x_ind][y_ind][z_ind][1] = grad_curr.x;
                    corner_data[x_ind][y_ind][z_ind][2] = grad_curr.y;
                    corner_data[x_ind][y_ind][z_ind][3] = grad_curr.z;

                }
            }
        }
        get_bicubic_coeff(coeff,corner_data);
    }

    for (int i = 0; i<4; i++){
        for (int j =0;j<4; j++){
            for (int k = 0;k<4; k++){
                coeff_curr = coeff[i][j][k];

                ind = k + 4*j + 4*4*i + 4*4*4*x0 + y0*vol->gridx*64 + z0*vol->gridx*vol->gridy*64;
                vol->tricubic_coeff[ind]=coeff_curr;

            }
        }
    }    

}

void 
calculate_grads_at_index(const VRVOL* vol,int x, int y, int z){

    int idx_less;
    int idx_more;
    float grad_curr;
    
    int idx_list[] = {x,y,z};
    size_t size_list[] = {vol->gridx,vol->gridy,vol->gridz};
    // glm::vec3 grad_total = glm::vec3(0.,0.,0.);
    int ind;

    for (int curr_dim=0; curr_dim<3; curr_dim++){

        idx_less = idx_list[curr_dim] - 1;
        idx_more = idx_list[curr_dim] + 1;

        if (idx_less<0){
            idx_less = idx_list[curr_dim];
        }

        if (idx_more>=size_list[curr_dim]){
            idx_more = idx_list[curr_dim];
        }
        
        // (idx_less < 0) ? (idx_less = idx_list[curr_dim]);
        // (idx_more > size_list[curr_dim]) ? (idx_more = idx_list[curr_dim]);

        if (curr_dim==0){
            grad_curr = (-1.*get_val_at_index(vol, idx_less, y, z)+ get_val_at_index(vol, idx_more, y, z))/2.;
            // grad_total.x = grad_curr;
        }
        else if (curr_dim==1){
            grad_curr = (-1.*get_val_at_index(vol, x, idx_less, z)+ get_val_at_index(vol, x, idx_more, z))/2.;
            // grad_total.y = grad_curr;
        }
        else{
            grad_curr = (-1.*get_val_at_index(vol, x, y, idx_less)+ get_val_at_index(vol, x, y, idx_more))/2.;
            // grad_total.z = grad_curr;    
        }

        ind = x*3;
        ind += y*vol->gridx*3;
        ind += z*vol->gridx*vol->gridy*3;
        ind = ind+ curr_dim;
        ((float*)vol->gx)[ind] = grad_curr;

        // if (x==5 && y==6 && z==14){
        //     EPRINT("in calc %d,%d %d %d %f %f %f\n",ind,x,y,z,((float*)vol->gx)[ind],((float*)vol->gx)[ind+1],((float*)vol->gx)[ind+2]);
        //     break;
        // }
    }


    // return grad_total;

}

glm::vec3
get_grads_at_index(const VRVOL* vol,int x, int y, int z){

    // int ind = x;
    // ind += x+y*vol->gridx;
    // ind += z*vol->gridx*vol->gridy;
    int ind = 3*x + y*vol->gridx*3 + z*vol->gridx*vol->gridy*3;

    float gx_curr = ((float*)vol->gx)[ind];
    float gy_curr = ((float*)vol->gx)[ind+1];
    float gz_curr = ((float*)vol->gx)[ind+2];
    
    // if (x==5 && y==6 && z==14){
    //     // EPRINT( " in get idx %d %d %d %f %f %f\n",x,y,z,((float*)vol->gx)[ind],((float*)vol->gx)[ind],((float*)vol->gx)[ind]);
    //     EPRINT( " in get idx %d %d %d %f %f %f\n",x,y,z,((float*)vol->gx)[ind],((float*)vol->gx)[ind+1],((float*)vol->gx)[ind+2]);
    //     // EPRINT( " in get idx %d %d %d %d %d %d\n",x,y,z,((uint8_t*)vol->data)[ind],((uint8_t*)vol->data)[ind],((uint8_t*)vol->data)[ind]);
    // }
    
    glm::vec3 grad_curr = glm::vec3(gx_curr,gy_curr,gz_curr);

    return grad_curr;

}


float
interpolate_trilinear_ui8(const VRVOL* vol, glm::vec3 pt)
{
    if(pt.x < 0 || pt.y < 0 || pt.z < 0 ||
       pt.x >= vol->gridx || pt.y >= vol->gridy || pt.z >= vol->gridz)
    {
        return 0.0; //this treats everything outside the volume as 0
                    //there are other choices, but this will work for
                    //most cases.
    }

    float x0  = floorf(pt.x);
    float x1  = ceilf(pt.x);

    float y0  = floorf(pt.y);
    float y1  = ceilf(pt.y);
    
    float z0  = floorf(pt.z);
    float z1  = ceilf(pt.z);
    
    float xd = pt.x - x0;
    float yd = pt.y - y0;
    float zd = pt.z - z0;

    float c000 = get_val_at_index(vol,(int)x0,int(y0),int(z0));
    float c001 = get_val_at_index(vol,(int)x0,int(y0),int(z1));
    float c010 = get_val_at_index(vol,(int)x0,int(y1),int(z0));
    float c100 = get_val_at_index(vol,(int)x1,int(y0),int(z0));
    float c110 = get_val_at_index(vol,(int)x1,int(y1),int(z0));
    float c011 = get_val_at_index(vol,(int)x0,int(y1),int(z1));
    float c101 = get_val_at_index(vol,(int)x1,int(y0),int(z1));
    float c111 = get_val_at_index(vol,(int)x1,int(y1),int(z1));

    float c00 = c000 * (1. - xd) + c100 * xd;
    float c01 = c001 * (1. - xd) + c101 * xd;
    float c10 = c010 * (1. - xd) + c110 * xd;
    float c11 = c011 * (1. - xd) + c111 * xd;

    float c0 = c00 * (1. - yd) + c10 * yd;
    float c1 = c01 * (1. - yd) + c11 * yd;

    float val = c0 * (1. - zd) + c1 * zd;
    val = val/MAX_UI8;
    
    // learned from wikipedia triliear interpolation page
    // https://en.wikipedia.org/wiki/Trilinear_interpolation

    return val;
}

float
mul_and_add(float a[], float b[], float c[]){
    float out = 0.;
    for (int i=0;i<8;i++){
        out = out + a[i]*b[i]*c[i];
    }
    return out;
}



glm::vec3
gradient_trilinear_ui8(const VRVOL*vol, glm::vec3 pt)
{
    //we can do this more effieinctly by explicitly looking up the data, but
    //this is a better representation.
    glm::vec3 grad;

    float x0  = floorf(pt.x);
    float x1  = ceilf(pt.x);

    float y0  = floorf(pt.y);
    float y1  = ceilf(pt.y);
    
    float z0  = floorf(pt.z);
    float z1  = ceilf(pt.z);
    
    float c000 = get_val_at_index(vol,(int)x0,int(y0),int(z0));
    float c001 = get_val_at_index(vol,(int)x0,int(y0),int(z1));
    float c010 = get_val_at_index(vol,(int)x0,int(y1),int(z0));
    float c100 = get_val_at_index(vol,(int)x1,int(y0),int(z0));
    float c110 = get_val_at_index(vol,(int)x1,int(y1),int(z0));
    float c011 = get_val_at_index(vol,(int)x0,int(y1),int(z1));
    float c101 = get_val_at_index(vol,(int)x1,int(y0),int(z1));
    float c111 = get_val_at_index(vol,(int)x1,int(y1),int(z1));


    float c_vals[] = {-1. * c000, +1. * c001, +1. * c010,
                      -1. * c011, +1. * c100, -1. * c101,
                      -1. * c110, +1. * c111};
    float x_vals[] = {x1, x1, x1, x1, x0, x0, x0, x0};
    float y_vals[] = {y1, y1, y0, y0, y1, y1, y0, y0};
    float z_vals[] = {z1, z0, z1, z0, z1, z0, z1, z0};
    float ones[] = {1., 1., 1., 1., 1., 1., 1., 1.};

    float a1 = mul_and_add(c_vals,y_vals,z_vals);
    float a2 = mul_and_add(c_vals,x_vals,z_vals);
    float a3 = mul_and_add(c_vals,x_vals,y_vals);
    float a4 = -1 * mul_and_add(c_vals,z_vals,ones);
    float a5 = -1 * mul_and_add(c_vals,y_vals,ones);
    float a6 = -1 * mul_and_add(c_vals,x_vals,ones);
    float a7 = mul_and_add(c_vals,ones,ones);


    grad.x = a1 + a4 * pt.y + a5 * pt.z + a7 * pt.y * pt.z;
    grad.y = a2 + a4 * pt.x + a6 * pt.z + a7 * pt.x * pt.z;
    grad.x = a3 + a5 * pt.x + a6 * pt.y + a7 * pt.x * pt.y;




    // grad.x = interpolate_nearest_ui8(vol,pt+glm::vec3(1,0,0))-
    //             interpolate_nearest_ui8(vol,pt-glm::vec3(1,0,0));
    // grad.y = interpolate_nearest_ui8(vol,pt+glm::vec3(0,1,0))-
    //             interpolate_nearest_ui8(vol,pt-glm::vec3(0,1,0));
    // grad.z = interpolate_nearest_ui8(vol,pt+glm::vec3(0,0,1))-
    //             interpolate_nearest_ui8(vol,pt-glm::vec3(0,0,1));
    return grad;
}


glm::vec3
gradient_trilinear_try2_ui8(const VRVOL*vol, glm::vec3 pt)
{
    if(pt.x < 0 || pt.y < 0 || pt.z < 0 ||
       pt.x >= vol->gridx || pt.y >= vol->gridy || pt.z >= vol->gridz)
    {
        return glm::vec3(0.0,0.0,0.0); //this treats everything outside the volume as 0
                    //there are other choices, but this will work for
                    //most cases.
    }

    float x0  = floorf(pt.x);
    float x1  = ceilf(pt.x);

    float y0  = floorf(pt.y);
    float y1  = ceilf(pt.y);
    
    float z0  = floorf(pt.z);
    float z1  = ceilf(pt.z);
    
    float xd = pt.x - x0;
    float yd = pt.y - y0;
    float zd = pt.z - z0;
    float diff;

    glm::vec3 c000 = get_grads_at_index(vol,(int)x0,int(y0),int(z0));
    // glm::vec3 c000_check = get_grads_at_index(vol,(int)x0,int(y0),int(z0));

    glm::vec3 c001 = get_grads_at_index(vol,(int)x0,int(y0),int(z1));
    glm::vec3 c010 = get_grads_at_index(vol,(int)x0,int(y1),int(z0));
    glm::vec3 c100 = get_grads_at_index(vol,(int)x1,int(y0),int(z0));
    glm::vec3 c110 = get_grads_at_index(vol,(int)x1,int(y1),int(z0));
    glm::vec3 c011 = get_grads_at_index(vol,(int)x0,int(y1),int(z1));
    glm::vec3 c101 = get_grads_at_index(vol,(int)x1,int(y0),int(z1));
    glm::vec3 c111 = get_grads_at_index(vol,(int)x1,int(y1),int(z1));

    // if (x0==5. && y0==6. && z0==14.)
    // {
    //     EPRINT("wroks %f %f %f %f %f %f\n",x0,y0,z0,c000.x,c000.y,c000.z);
    //     EPRINT("%f %f %f %f %f %f\n",x0,y0,z0,c000_check.x,c000_check.y,c000_check.z);

    // }

    diff = 1.-xd;
    
    glm::vec3 c00 = c000 * diff + c100 * xd;
    glm::vec3 c01 = c001 * diff + c101 * xd;
    glm::vec3 c10 = c010 * diff + c110 * xd;
    glm::vec3 c11 = c011 * diff + c111 * xd;

    diff = 1. - yd;
    glm::vec3 c0 = c00 * diff + c10 * yd;
    glm::vec3 c1 = c01 * diff + c11 * yd;

    diff = 1. - zd;
    glm::vec3 val = c0 * diff + c1 * zd +epsilon;

    // val = val/MAX_UI8;
    
    // // learned from wikipedia triliear interpolation page
    // // https://en.wikipedia.org/wiki/Trilinear_interpolation

    return val;
}

glm::vec3
gradient_tricubic_ui8(const VRVOL*vol, glm::vec3 pt)
{   

    if(pt.x < 0 || pt.y < 0 || pt.z < 0 ||
       pt.x >= vol->gridx || pt.y >= vol->gridy || pt.z >= vol->gridz)
    {
        return glm::vec3(0.0,0.0,0.0); //this treats everything outside the volume as 0
                    //there are other choices, but this will work for
                    //most cases.
    }

    int x0  = (int) floorf(pt.x);
    int x1  = (int) ceilf(pt.x);

    int y0  = (int) floorf(pt.y);
    int y1  = (int) ceilf(pt.y);
    
    int z0  = (int) floorf(pt.z);
    int z1  = (int) ceilf(pt.z);

    float xd = pt.x - floorf(pt.x);
    float yd = pt.y - floorf(pt.y);
    float zd = pt.z - floorf(pt.z);

    int i,j,k;
    float gradx, grady, gradz;
    glm::vec3 grad_total;


    float coeff[4][4][4];
    for (i=0;i<4;i++){
        for (j=0;j<4;j++){
            for (k=0;k<4;k++){
                coeff[i][j][k] = get_bicubic_coeff_at_index(vol,x0,y0,z0,i,j,k);
            }
        }
    }

    gradx = 0;
    for (i=0; i<3; i++){
        for (j=0; j<4; j++){
            for (k=0; k<4; k++){
                gradx += (coeff[i+1][j][k] - coeff[i][j][k]) * 
                        (nCr_2[i] * pow(1. - xd, 2-i) * pow(xd, i)) *
                        (nCr[j] * pow(1. - yd, 3-j) * pow(yd, j))*
                        (nCr[k] * pow(1. - zd, 3-k) * pow(zd, k));
            }
        }
    }

    grady = 0;
    for (i=0; i<4; i++){
        for (j=0; j<3; j++){
            for (k=0; k<4; k++){
                grady += (coeff[i][j+1][k] - coeff[i][j][k]) * 
                        (nCr[i] * pow(1. - xd, 3-i) * pow(xd, i)) *
                        (nCr_2[j] * pow(1. - yd, 2-j) * pow(yd, j))*
                        (nCr[k] * pow(1. - zd, 3-k) * pow(zd, k));
            }
        }
    }

    gradz = 0;
    for (i=0; i<4; i++){
        for (j=0; j<4; j++){
            for (k=0; k<3; k++){
                gradz += (coeff[i][j][k+1] - coeff[i][j][k]) * 
                        (nCr[i] * pow(1. - xd, 3-i) * pow(xd, i)) *
                        (nCr[j] * pow(1. - yd, 3-j) * pow(yd, j))*
                        (nCr_2[k] * pow(1. - zd, 2-k) * pow(zd, k));
            }
        }
    }

    grad_total = glm::vec3(gradx,grady,gradz) * 3.f +epsilon;
    return grad_total;
    // glm::vec3(0);
}

void 
get_bicubic_coeff(float coeff[4][4][4],float corner_data[2][2][2][4]){
    
    float mini_coeff[4] = {0,1./3.,-1*1./3.,0};
    int coord_to_choose[4] = {0,0,1,1};

    float f_val;
    float g_x;
    float g_y;
    float g_z;

    for (int i=0;i<4;i++){
        for (int j=0;j<4;j++){
            for (int k=0;k<4;k++){
                f_val = corner_data[coord_to_choose[i]][coord_to_choose[j]][coord_to_choose[k]][0];
                g_x = corner_data[coord_to_choose[i]][coord_to_choose[j]][coord_to_choose[k]][1];
                g_y = corner_data[coord_to_choose[i]][coord_to_choose[j]][coord_to_choose[k]][2];
                g_z = corner_data[coord_to_choose[i]][coord_to_choose[j]][coord_to_choose[k]][3];
                coeff[i][j][k] = f_val + mini_coeff[i]*g_x + mini_coeff[j]*g_y + mini_coeff[k]*g_z;
            }
        }
    }

}

float get_bicubic_coeff_at_index(const VRVOL *vol, int x, int y, int z, int i, int j, int k){

    int ind = k + 4*j + 4*4*i + 4*4*4*x + y*vol->gridx*64 + z*vol->gridx*vol->gridy*64;
    return vol->tricubic_coeff[ind];
}

float
interpolate_tricubic_ui8(const VRVOL* vol, glm::vec3 pt)
{
    if(pt.x < 0 || pt.y < 0 || pt.z < 0 ||
       pt.x >= vol->gridx || pt.y >= vol->gridy || pt.z >= vol->gridz)
    {
        return 0.0; //this treats everything outside the volume as 0
                    //there are other choices, but this will work for
                    //most cases.
    }


    int x0  = (int) floorf(pt.x);
    // int x1  = (int) ceilf(pt.x);

    int y0  = (int) floorf(pt.y);
    // int y1  = (int) ceilf(pt.y);
    
    int z0  = (int) floorf(pt.z);
    // int z1  = (int) ceilf(pt.z);

    float xd = pt.x - floorf(pt.x);
    float yd = pt.y - floorf(pt.y);
    float zd = pt.z - floorf(pt.z);
    
    // int coord[3][2]={  {x0,x1},{y0,y1},{z0,z1}};
    // // int coord[3][2]={  {0,1},{0,1},{0,1}};

    // //make cubes
    // float corner_data[2][2][2][4];
    // float f_curr;
    // glm::vec3 grad_curr;

    // for (int x_ind=0;x_ind<2;x_ind++){
    //     for (int y_ind=0;y_ind<2;y_ind++){
    //         for (int z_ind=0;z_ind<2;z_ind++){

    //             f_curr = get_val_at_index(vol,coord[0][x_ind],coord[1][y_ind],coord[2][z_ind]);
    //             grad_curr = get_grads_at_index(vol,coord[0][x_ind],coord[1][y_ind],coord[2][z_ind]);
    //             corner_data[x_ind][y_ind][z_ind][0] = f_curr;
    //             corner_data[x_ind][y_ind][z_ind][1] = grad_curr.x;
    //             corner_data[x_ind][y_ind][z_ind][2] = grad_curr.y;
    //             corner_data[x_ind][y_ind][z_ind][3] = grad_curr.z;

    //         }
    //     }
    // }

    float coeff[4][4][4];
    
    for (int i=0;i<4;i++){
        for (int j=0;j<4;j++){
            for (int k=0;k<4;k++){
                coeff[i][j][k] = get_bicubic_coeff_at_index(vol,x0,y0,z0,i,j,k);
            }
        }
    }

    // get_bicubic_coeff(coeff,corner_data);

    // float mini_coeff[4] = {0,1./3.,-1*1./3.,0};
    // int coord_to_choose[4] = {0,0,1,1};

    // float f_val;
    // float g_x;
    // float g_y;
    // float g_z;

    // for (int i=0;i<4;i++){
    //     for (int j=0;j<4;j++){
    //         for (int k=0;k<4;k++){
    //             f_val = corner_data[coord_to_choose[i]][coord_to_choose[j]][coord_to_choose[k]][0];
    //             g_x = corner_data[coord_to_choose[i]][coord_to_choose[j]][coord_to_choose[k]][1];
    //             g_y = corner_data[coord_to_choose[i]][coord_to_choose[j]][coord_to_choose[k]][2];
    //             g_z = corner_data[coord_to_choose[i]][coord_to_choose[j]][coord_to_choose[k]][3];
    //             coeff[i][j][k] = f_val + mini_coeff[i]*g_x + mini_coeff[j]*g_y + mini_coeff[k]*g_z;
    //         }
    //     }
    // }

    

    
    

    float val = 0;

    for (int i=0;i<4;i++){
        for (int j=0;j<4;j++){
            for (int k=0;k<4;k++){

                val += coeff[i][j][k] *
                // val += get_bicubic_coeff_at_index(vol,x0,y0,z0,i,j,k) * 
                        (nCr[i] * pow(1. - xd, 3-i) * pow(xd, i)) *
                        (nCr[j] * pow(1. - yd, 3-j) * pow(yd, j))*
                        (nCr[k] * pow(1. - zd, 3-k) * pow(zd, k));
                
                // if (x0==2 && y0==1 && z0==0){
                //     EPRINT("%d %d %d %f %f\n",i,j,k,coeff[i][j][k],get_bicubic_coeff_at_index(vol,x0,y0,z0,i,j,k));
                // }
            }
        }
    }
    val = val/MAX_UI8;

    

    return val;
}


/*
 * Returns the gradient at pt, this is not the most efficient
 * way to implement this function, but it will work.
 */
glm::vec3
gradient_nearest_ui8(const VRVOL*vol, glm::vec3 pt)
{
    //we can do this more effieinctly by explicitly looking up the data, but
    //this is a better representation.
    glm::vec3 grad;
    grad.x = interpolate_nearest_ui8(vol,pt+glm::vec3(1,0,0))-
                interpolate_nearest_ui8(vol,pt-glm::vec3(1,0,0));
    grad.y = interpolate_nearest_ui8(vol,pt+glm::vec3(0,1,0))-
                interpolate_nearest_ui8(vol,pt-glm::vec3(0,1,0));
    grad.z = interpolate_nearest_ui8(vol,pt+glm::vec3(0,0,1))-
                interpolate_nearest_ui8(vol,pt-glm::vec3(0,0,1));
    return grad + epsilon;
}


int
vrv_initvolume(VRVOL *vol,
               void *data,
               void *gx,
               float *tri_in,
               uint32_t grid_x,
               uint32_t grid_y,
               uint32_t grid_z,
               glm::vec3 vdim,
               VINTERP_T interpolation,
               VOL_T data_type)
{
    if(!data)
    {
        EPRINT("ERROR: Null Data passed to vrc_setvolume\n");
        return 0;
    }
    if(vdim.x <= 0|| vdim.y <=0||vdim.z<=0)
    {
        EPRINT("ERROR: Bad Volume Dimensions w: %f h: %f d: %f\n",vdim.x,vdim.y,vdim.z);
        return 0;
    }

    //compute bounding box
    vol->bb_p0 = glm::vec3(0.0,0.0,0.0);
    vol->bb_p1 = vdim;
    if(vdim.x > 1|| vdim.y > 1 || vdim.z > 1)
    {
        EPRINT("WARNING: Bad Volume Dimensions w: %f h: %f d: %f\n",vdim.x,vdim.y,vdim.z);
        vol->bb_p1 = glm::normalize(vol->bb_p1);
        EPRINT("WARNING: Normalizing to w: %f h: %f d: %f\n",vol->bb_p1.x,vol->bb_p1.y,vol->bb_p1.z);
    }

    //center at <0>
    vol->bb_p0 = -vol->bb_p1* .5f;
    vol->bb_p1 = vol->bb_p1* .5f;

    vol->inv_size = 1.0f/(vol->bb_p1-vol->bb_p0);

    //save data information
    vol->coeff_done = false;
    vol->data = data;
    vol->gx = gx;
    vol->tricubic_coeff = tri_in;
    vol->gridx = grid_x;
    vol->gridy = grid_y;
    vol->gridz = grid_z;

    vol->type = data_type;
    vol->interp = interpolation;

    return 1;

}

void
vrv_initgradient(VRVOL *vol)
{
    for (int z=0; z<vol->gridz;z++){    
        for (int y=0; y<vol->gridy;y++){
            for (int x=0; x<vol->gridx;x++){            
                calculate_grads_at_index(vol,x,y,z);
            }   
        }       
    }
}

void
vrv_initcoeff(VRVOL *vol)
{   

    for (int z=0; z<vol->gridz;z++){    
        for (int y=0; y<vol->gridy;y++){
            for (int x=0; x<vol->gridx;x++){            

                calculate_coeff_at_index(vol,x,y,z);
            }   
        }       
    }
}

VRVOL* readvol_ui8(const char *fn, size_t gridx, size_t gridy, size_t gridz, glm::vec3 voldim, VINTERP_T interpolation)
{
    FILE *fd;
    fd = fopen(fn,"rb");
    if(!fd){
        EPRINT("Error opening file %s\n",fn);
    }

    uint8_t *in = (uint8_t *)malloc(gridx*gridy*gridz);
    float *gx_in = (float *)malloc(gridx*gridy*gridz*3*sizeof(float));

    float *tri_in = (float *)malloc(gridx*gridy*gridz*4*4*4*sizeof(float));
    
    if(fread(in,1,gridx*gridy*gridz,fd) < gridx*gridy*gridz){
        free(in);
        EPRINT("Error reading in volume %s\n",fn);
    }
    fclose(fd);

    VRVOL *vol = (VRVOL*)malloc(sizeof(VRVOL));
    // vol->gx = gx;
    if(!vrv_initvolume(vol,in,gx_in,tri_in,gridx,gridy,gridz,voldim,interpolation,VRV_UINT8))
    {
        EPRINT("Error creating volume structure\n");
        free(in);
        free(gx_in);
        free(tri_in);
        free(vol);
        // free(gy);
        // free(gz);
        return NULL;
    }

    // vrv_initgradient(vol,gx,gy,gz);
    vrv_initgradient(vol);
    if (interpolation==VI_CUBIC){
        vrv_initcoeff(vol);
        vol->coeff_done=true;
    }

    return vol;
}

VRVOL* readvol_ui8_sphere(size_t gridx, size_t gridy, size_t gridz, glm::vec3 voldim, VINTERP_T interpolation)
{
 
    uint8_t *in = (uint8_t *)malloc(gridx*gridy*gridz);

    float *gx_in = (float *)malloc(gridx*gridy*gridz*3*sizeof(float));
    
    float *tri_in = (float *)malloc(gridx*gridy*gridz*4*4*4*sizeof(float));


    float biggest= pow(pow(float(gridx),2.)+pow(float(gridy),2)+pow(float(gridz),2),0.5);
    float mult = float(MAX_UI8)/biggest;

    for (int z = 0;z<gridz;z++){
        for (int y = 0;y<gridy;y++){
            for (int x = 0;x<gridx;x++){
                int ind = x + y*gridx + z*gridx*gridy;
                int f = (int)roundf(mult*pow(pow(float(x),2.)+pow(float(y),2)+pow(float(z),2),0.5));
                EPRINT("%d \n",f);
                in[ind] = f;
            }
        }
    }
    
    voldim.x = gridx;
    voldim.y = gridy;
    voldim.z = gridz;

    VRVOL *vol = (VRVOL*)malloc(sizeof(VRVOL));
    if(!vrv_initvolume(vol,in,gx_in,tri_in,gridx,gridy,gridz,voldim,interpolation,VRV_UINT8))
    {
        EPRINT("Error creating volume structure\n");
        free(in);
        free(gx_in);
        free(tri_in);
        free(vol);
        return NULL;
    }
    
    vrv_initgradient(vol);
    vrv_initcoeff(vol);
    

    return vol;
}

//ADD a new interpolation scheme by adding it to the
//VINTERP_T enum, for example add VI_LINEAR which will
//get the value 1<<8+1.  Then you can add that combination
//to the switch statment i.e.
//case VRV_UINT8 | VI_LINEAR
//
//Note if you want to support other volume types, say float
//you will need to implement an interpolation function for each
//interpolation type i.e.
//VRV_FLOAT32 | VI_NEAREST and VRV_FLOAT32 | VRV_LINEAR ... etc
//Most people will not need to do this, and it is not required for
//the class
float vrv_interpolate(const VRVOL* vol, glm::vec3 pt)
{
    switch(vol->type|vol->interp){
        case VRV_UINT8 | VI_NEAREST:
            return interpolate_nearest_ui8(vol,pt);
        case VRV_UINT8 | VI_LINEAR:
            return interpolate_trilinear_ui8(vol,pt);
        case VRV_UINT8 | VI_CUBIC:
            return interpolate_tricubic_ui8(vol,pt);
        
            // interpolate_tricubic_ui8(vol,pt);
            // 
        default:
            EPRINT("BAD VOLUME TYPE or INTERPOLATION METHOD\n");
            return 0;
    }
}

glm::vec3 vrv_gradient(const VRVOL* vol, glm::vec3 pt){
    switch(vol->type|vol->interp){
        case VRV_UINT8|VI_NEAREST:
            gradient_nearest_ui8(vol,pt);
        case VRV_UINT8|VI_LINEAR:
            return gradient_trilinear_try2_ui8(vol,pt);
        case VRV_UINT8|VI_CUBIC:
            return gradient_tricubic_ui8(vol,pt);
        default:
            EPRINT("BAD VOLUME IN TYPE or INTERPOLATION Combonation in Volume");
            return glm::vec3(0);
    }
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
