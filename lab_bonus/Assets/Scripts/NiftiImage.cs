using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;
using Unity.Mathematics;
using Debug = UnityEngine.Debug;

namespace Nifti {

    public sealed class NiftiImage : IEnumerable {
        
        public NiftiHeader Header { get; private set; }    // stores meta-data of the file
        public dynamic[] Data { get; private set; }        // stores the actual voxel values
        public float3[] Coordinates { get; private set; }  // stores the 3D location of each voxel

        private bool _byteSwapping = false;  // whether the data needs to be byte swapped

        // constructor
        public NiftiImage(string filepath) {
            if (!File.Exists(filepath)) {
                throw new FileNotFoundException("Cannot find file: ", filepath);
            }

            if (Path.GetExtension(filepath) != ".nii") {
                // here we only support the uncompressed version, so .nii.gz is not allowed
                throw new FormatException("Incorrect file extension, expected .nii");
            }

            // load header and data while the filestream is alive
            using (var reader = new BinaryReader(new FileInfo(filepath).OpenRead())) {
                LoadHeader(reader);
                LoadData(reader);
            }

            // compute the spatial coordinate (x,y,z) of each voxel based on the array index (i,j,k)
            CalculateCoordinates();
        }

        private void LoadHeader(BinaryReader reader) {
            Header = new NiftiHeader();
            
            // start reading header
            Header.sizeof_hdr = reader.ReadInt32();
            Header.data_type = reader.ReadBytes(10);
            Header.db_name = reader.ReadBytes(18);
            Header.extents = reader.ReadInt32();
            Header.session_error = reader.ReadInt16();
            Header.regular = reader.ReadByte();
            Header.dim_info = reader.ReadByte();

            for (var i = 0; i < 8; i++) { Header.dim[i] = reader.ReadInt16(); }

            Header.intent_p1 = reader.ReadSingle();
            Header.intent_p2 = reader.ReadSingle();
            Header.intent_p3 = reader.ReadSingle();
            Header.intent_code = reader.ReadInt16();
            Header.datatype = reader.ReadInt16();
            Header.bitpix = reader.ReadInt16();
            Header.slice_start = reader.ReadInt16();

            for (var i = 0; i < 8; i++) { Header.pixdim[i] = reader.ReadSingle(); }

            Header.vox_offset = reader.ReadSingle();
            Header.scl_slope = reader.ReadSingle();
            Header.scl_inter = reader.ReadSingle();
            Header.slice_end = reader.ReadInt16();
            Header.slice_code = reader.ReadByte();
            Header.xyzt_units = reader.ReadByte();
            Header.cal_max = reader.ReadSingle();
            Header.cal_min = reader.ReadSingle();
            Header.slice_duration = reader.ReadSingle();
            Header.toffset = reader.ReadSingle();
            Header.glmax = reader.ReadInt32();
            Header.glmin = reader.ReadInt32();
            Header.descrip = reader.ReadBytes(80);
            Header.aux_file = reader.ReadBytes(24);
            Header.qform_code = reader.ReadInt16();
            Header.sform_code = reader.ReadInt16();
            Header.quatern_b = reader.ReadSingle();
            Header.quatern_c = reader.ReadSingle();
            Header.quatern_d = reader.ReadSingle();
            Header.qoffset_x = reader.ReadSingle();
            Header.qoffset_y = reader.ReadSingle();
            Header.qoffset_z = reader.ReadSingle();

            for (var i = 0; i < 4; i++) { Header.srow_x[i] = reader.ReadSingle(); }
            for (var i = 0; i < 4; i++) { Header.srow_y[i] = reader.ReadSingle(); }
            for (var i = 0; i < 4; i++) { Header.srow_z[i] = reader.ReadSingle(); }

            Header.intent_name = reader.ReadBytes(16);
            Header.magic = reader.ReadBytes(4);
            Header.extension = reader.ReadBytes(4);
            // finish reading header, total: 352 bytes
            
            ValidateHeader();  // validate header information
            
            // if endianness of the host CPU architecture is not compatible with the .nii file,
            // file data must be byte-swapped in order to maintain data integrity
            if (Header.dim[0] < 0 || Header.dim[0] > 7) {
                _byteSwapping = true;
                ConvertEndianness();
            }
        }

        private void LoadData(BinaryReader reader) {
            // initialize the data array (as a one-dimensional array)
            Data = new dynamic[Header.dim[1] * Header.dim[2] * Header.dim[3]];
            
            // make sure our binary reader is currently at the correct offset position
            if (reader.BaseStream.Position != (long) Header.vox_offset) {
                Debug.LogException(new IOException("Corrupted offset pointer!"));
                return;
            }
            
            // https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/dim.html
            // the byte order of the data arrays is presumed to be the same as the byte order of the header
            for (var i = 0; i < Data.Length; i++) {
                if (!_byteSwapping) {
                    if (Header.datatype == (ushort) NiftiType.Float32) {
                        Data[i] = reader.ReadSingle();
                    }
                    else {
                        Data[i] = reader.ReadDouble();
                    }
                }
                else {
                    if (Header.datatype == (ushort) NiftiType.Float32) {
                        var bytes = BitConverter.GetBytes(reader.ReadSingle());
                        Array.Reverse(bytes);
                        Data[i] = BitConverter.ToSingle(bytes, 0);
                    }
                    else {
                        var bytes = BitConverter.GetBytes(reader.ReadDouble());
                        Array.Reverse(bytes);
                        Data[i] = BitConverter.ToDouble(bytes, 0);
                    }
                }
            }

            // after all data has been read, make sure we are at the end of the file stream
            if (reader.BaseStream.Position != reader.BaseStream.Length) {
                Debug.LogException(new IOException("Corrupted file data, filestream has not reached the end!"));
            }

            // https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/scl_slopeinter.html
            // if the scl_slope field is nonzero, each voxel value should be scaled as necessary
            if (Header.scl_slope != 0 && !Mathf.Approximately(Header.scl_slope, 1.0f)) {
                var slope = Header.scl_slope;
                var intercept = Header.scl_inter;
                
                for (var i = 0; i < Data.Length; i++) {
                    Data[i] = slope * Data[i] + intercept;
                }
            }
        }
        
        private void SwapBytes(FieldInfo field) {
            if (field.FieldType.IsArray) {
                var array = field.GetValue(Header) as Array;
                var elementType = array?.GetType().GetElementType();

                if (elementType == typeof(short)) {
                    var reversedList = new List<short>();
                    foreach (var value in array) {
                        var bytes = BitConverter.GetBytes((short) value);
                        Array.Reverse(bytes, 0, bytes.Length);
                        reversedList.Add(BitConverter.ToInt16(bytes, 0));
                    }

                    field.SetValue(Header, reversedList.ToArray());
                }

                else if (elementType == typeof(float)) {
                    var reversedList = new List<float>();
                    foreach (var value in array) {
                        var bytes = BitConverter.GetBytes((float) value);
                        Array.Reverse(bytes, 0, bytes.Length);
                        reversedList.Add(BitConverter.ToSingle(bytes, 0));
                    }

                    field.SetValue(Header, reversedList.ToArray());
                }

                else {
                    throw new InvalidDataException("Invalid element type in header.");
                }
            }

            else {
                var value = field.GetValue(Header);

                if (field.FieldType == typeof(int)) {
                    var bytes = BitConverter.GetBytes((int) value);
                    Array.Reverse(bytes, 0, bytes.Length);
                    field.SetValue(Header, BitConverter.ToInt32(bytes, 0));
                }
                else if (field.FieldType == typeof(short)) {
                    var bytes = BitConverter.GetBytes((short) value);
                    Array.Reverse(bytes, 0, bytes.Length);
                    field.SetValue(Header, BitConverter.ToInt16(bytes, 0));
                }
                else if (field.FieldType == typeof(float)) {
                    var bytes = BitConverter.GetBytes((float) value);
                    Array.Reverse(bytes, 0, bytes.Length);
                    field.SetValue(Header, BitConverter.ToSingle(bytes, 0));
                }
                else {
                    throw new InvalidDataException("Invalid field type in header.");
                }
            }
        }
        
        private void ConvertEndianness() {
            // https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/dim.html
            // use C# reflection to convert endianness in each header field (by reversing the bytes array)
            var fields = typeof(NiftiHeader).GetFields();
            
            foreach (var field in fields) {
                if (field.FieldType != typeof(byte) && field.FieldType != typeof(byte[])) {
                    SwapBytes(field);
                }
            }
        }
        
        private void ValidateHeader() {
            // https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields
            if (Encoding.UTF8.GetString(Header.magic) != "n+1\0") {
                Debug.LogException(new FormatException("File does not conform with the nifti standard!"));
            }
            
            Debug.Log("Validating nifti header information...");
            
            Debug.Assert(Header.sizeof_hdr == 348, "Incorrect header size!");
            Debug.Assert(Header.slice_start <= Header.slice_end, "");
            Debug.Assert(Header.dim[0] == 3, "Invalid dim[0], only 3D image is supported!");
            Debug.Assert(Header.dim[1] > 0 && Header.dim[2] > 0 && Header.dim[3] > 0, "Invalid dimensions!");
            Debug.Assert(Header.intent_code == 0, "Unexpected intent code!");
            Debug.Assert(Mathf.Approximately(Header.vox_offset, 352.0f), "Byte offset != 352!");
            Debug.Assert(Header.qform_code > 0, "Compatibility with ANALYZE 7.5 format is not supported!");

            // information about the location of the voxels in some standard space may be present but will be ignored
            // we only care about the nominal voxel locations as reported by the scanner, rather than registration
            if (Header.sform_code > 0) {
                Debug.LogWarning("Standard space information is present but will be ignored...");
            }

            // we are only going to deal with float or double data types
            if (Header.datatype != (ushort)NiftiType.Float32 && Header.datatype != (ushort)NiftiType.Float64) {
                Debug.LogError("Only float or double data type is supported!");
            }
            
            if (Header.bitpix != 32 && Header.bitpix != 64) {
                Debug.LogError("Bits per voxel doesn't match data type!");
            }

            // pixdim[0] should be either -1 or 1, any other value is treated as 1
            if (!Mathf.Approximately(Mathf.Abs(Header.pixdim[0]), 1.0f)) {
                Header.pixdim[0] = 1.0f;
            }
            
            if (Header.extension[0] != 0) {
                Debug.LogWarning("Extension has been detected but will be silently ignored...");
            }

            Debug.Log($"Validation check passed!");
            
            // print out the meta-information to the console for verification purposes
            // this can help us get a sense of the underlying data and possibly spot any errors
            
            Debug.Log($"Data has {Header.dim[0]} dimensions with shape ({Header.dim[1]},{Header.dim[2]}" +
                      $",{Header.dim[3]}), which is stored as {((NiftiType) Header.datatype).ToString()} type," +
                      $" and has {Header.bitpix} bits per voxel.\nEach voxel is ({Header.pixdim[1]}" +
                      $",{Header.pixdim[2]},{Header.pixdim[3]}) units wide, the values stored in each voxel should" +
                      $" be linearly scaled by a slope of {Header.scl_slope} and an intercept of {Header.scl_inter}.");
            
            Debug.Log($"Original file description: {Encoding.UTF8.GetString(Header.descrip)}.");
            
            Debug.Log($"The total size of the image data is (in number of bytes): " +
                      $"{Header.dim[0] * Header.dim[1] * Header.dim[2] * Header.bitpix / 8}");
        }

        private void CalculateCoordinates() {
            // https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/qsform.html
            // a right-handed coordinate system is assumed: (+x = right, +y = anterior, +z = superior)
            
            // initialize the coordinates array (as a one-dimensional array)
            Coordinates = new float3[Header.dim[1] * Header.dim[2] * Header.dim[3]];
            
            // retrieve the quaternion numbers
            var b = Header.quatern_b;
            var c = Header.quatern_c;
            var d = Header.quatern_d;
            var a = Mathf.Sqrt(1 - b * b - c * c - d * d);
            
            // https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/quatern.html
            // rotation matrix (requires the Mathematics package from Unity)
            var rotation = new float3x3(
                new float3(a * a + b * b - c * c - d * d, 2 * (b * c + a * d), 2 * (b * d - a * c)),
                new float3(2 * (b * c - a * d), a * a - b * b + c * c - d * d, 2 * (a * b + c * d)),
                new float3(2 * (b * d + a * c), 2 * (c * d - a * b), a * a - b * b - c * c + d * d));

            // the quaternion factor, spacing and shift values
            var qfac = Header.pixdim[0];
            var spacing = new float3(Header.pixdim[1], Header.pixdim[2], Header.pixdim[3]);
            var shift = new float3(Header.qoffset_x, Header.qoffset_y, Header.qoffset_z);

            // take special care of the order of traversal (i runs faster, then j, then k)
            for (var k = 0; k < Header.dim[3]; k++) {           // z from inferior to superior
                for (var j = 0; j < Header.dim[2]; j++) {       // y from posterior to anterior
                    for (var i = 0; i < Header.dim[1]; i++) {   // x from left to right
                        // this is the location where voxel (i,j,k) is mapped into the array
                        var index = i + j * Header.dim[1] + k * Header.dim[1] * Header.dim[2];
                        // see explanation of method 2 on the nifti official website
                        Coordinates[index] = math.mul(rotation, new float3(i, j, qfac * k)) * spacing + shift;
                    }
                }
            }
        }

        // data array indexer (i, j, k)
        public dynamic this[int i, int j, int k] {
            get {
                // this is the location where voxel (i,j,k) is stored into the data array
                var index = i + j * Header.dim[1] + k * Header.dim[1] * Header.dim[2];

                try {
                    return Data[index];
                }
                catch (IndexOutOfRangeException e) {
                    Debug.LogError($"Invalid voxel index ({i},{j},{k}): {e.Message}");
                    return null;
                }
            }
        }

        // data array enumerator (allow the use of foreach loop on the data array)
        public IEnumerator GetEnumerator() => Data.GetEnumerator();
        
        // an indexer-like method to obtain the voxel coordinate
        public float3 GetVoxelCoordinate(int i, int j, int k) {
            var index = i + j * Header.dim[1] + k * Header.dim[1] * Header.dim[2];

            try {
                return Coordinates[index];
            }
            catch (IndexOutOfRangeException e) {
                Debug.LogError($"Invalid voxel index ({i},{j},{k}): {e.Message}");
                return new float3();
            }
        }
    }

    public static class NiftiImageTest {
        // uncomment the line below to run the test code in Unity
        // [RuntimeInitializeOnLoadMethod]
        public static void Main() {
            var dir = Directory.GetCurrentDirectory();
            var brain = new NiftiImage(Path.Combine(dir, "Assets", "Resources", "brain.nii"));
            
            // to keep it simple, here we use one-dimensional array to store all the voxels data & coordinates,
            // but the 1D array can be treated as if it were a 3D array that represents the 3D volume.
            // we have provided indexer methods that have easy access into both the data & coordinates array
            
            // the mapping relationship between a scalar index `idx` and an index tuple (i,j,k) is:
            // -----------------------------------------------------------------------------------
            //     i = idx % (Header.dim[1] * Header.dim[2]) % Header.dim[1];
            //     j = (int) ((idx % (Header.dim[1] * Header.dim[2])) / Header.dim[1]);
            //     k = (int) (idx / (Header.dim[1] * Header.dim[2]));
            // -----------------------------------------------------------------------------------
            //     idx = i + j * Header.dim[1] + k * Header.dim[1] * Header.dim[2];
            // -----------------------------------------------------------------------------------
            
            // the index number 11418536 and (134, 201, 108) are equivalent, both should return 44
            Debug.Log($"brain.Data[11418536] returns {brain.Data[11418536]}");  // direct access into the array
            Debug.Log($"brain[134, 201, 108] returns {brain[134, 201, 108]}");  // use indexer to access the array
            
            // use this method to get the world space position of a single voxel
            // the return value is a float3, which requires the Unity Mathematics package
            Debug.Log($"Voxel coordinate: {brain.GetVoxelCoordinate(134, 201, 108)}");
            
            // if an invalid index is specified, the program will throw the IndexOutOfRangeException and return null
        }
    }
}