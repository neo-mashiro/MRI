// ReSharper disable InconsistentNaming
// ReSharper disable IdentifierTypo
// ReSharper disable FieldCanBeMadeReadOnly.Global
// ReSharper disable CommentTypo

namespace Nifti {
    
    // https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/datatype.html
    public enum NiftiType : ushort {
        Float32 = 16,  // 32 bit float
        Float64 = 64   // 64 bit float = double
    }

    public class NiftiHeader {
        // https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields
        // total: 352 bytes
        public int     sizeof_hdr;                  // Must be 348                     0
        public byte[]  data_type = new byte[10];    // ++Unused++                      4
        public byte[]  db_name = new byte[18];      // ++Unused++                     14
        public int     extents;                     // ++Unused++                     32
        public short   session_error;               // ++Unused++                     36
        public byte    regular;                     // ++Unused++                     38
        public byte    dim_info;                    // MRI slice ordering             39
        public short[] dim = new short[8];          // Data array dimensions          40
        public float   intent_p1;                   // 1st intent parameter           56
        public float   intent_p2;                   // 2nd intent parameter           60
        public float   intent_p3;                   // 3rd intent parameter           64
        public short   intent_code;                 // NIFTIINTENT code               68
        public short   datatype;                    // Defines data type              70
        public short   bitpix;                      // Number bits/voxel              72
        public short   slice_start;                 // First slice index              74
        public float[] pixdim = new float[8];       // Grid spacings                  76
        public float   vox_offset;                  // Offset into .nii file         108
        public float   scl_slope;                   // Data scaling: slope           112
        public float   scl_inter;                   // Data scaling: offset          116
        public short   slice_end;                   // Last slice index              120
        public byte    slice_code;                  // Slice timing order            122
        public byte    xyzt_units;                  // Units of pixdim[1..4]         123
        public float   cal_max;                     // Max display intensity         124
        public float   cal_min;                     // Min display intensity         128
        public float   slice_duration;              // Time for 1 slice              132
        public float   toffset;                     // Time axis shift               136
        public int     glmax;                       // ++Unused++                    140
        public int     glmin;                       // ++Unused++                    144
        public byte[]  descrip = new byte[80];      // any text you like             148
        public byte[]  aux_file = new byte[24];     // auxiliary filename            228
        public short   qform_code;                  // NIFTIXFORM code               252
        public short   sform_code;                  // NIFTIXFORM code               254
        public float   quatern_b;                   // Quaternion b param            256
        public float   quatern_c;                   // Quaternion c param            260
        public float   quatern_d;                   // Quaternion d param            264
        public float   qoffset_x;                   // Quaternion x shift            268
        public float   qoffset_y;                   // Quaternion y shift            272
        public float   qoffset_z;                   // Quaternion z shift            276
        public float[] srow_x = new float[4];       // 1st row affine transform      280
        public float[] srow_y = new float[4];       // 2nd row affine transform      296
        public float[] srow_z = new float[4];       // 3rd row affine transform      312
        public byte[]  intent_name = new byte[16];  // name or meaning of data       328
        public byte[]  magic = new byte[4];         // Must be "ni1\0" or "n+1\0"    344
        public byte[]  extension = new byte[4];     // Header extension              348
    }
}