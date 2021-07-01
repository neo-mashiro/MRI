using System.Collections;
using NiftiLoader;
using NiftiProcessor;
using SFB;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace Gameplay {
    
    public class MenuController : MonoBehaviour {

        [SerializeField] private GameObject brain;
        [SerializeField] private GameObject loadingPanel;
        [SerializeField] private GameObject specPanel;

        [SerializeField] private Color brainColor;
        
        [SerializeField] private Button loadButton;
        [SerializeField] private Button segmentButton;
        [SerializeField] private Button meshButton;
        [SerializeField] private Button quitButton;
        [SerializeField] private Slider alphaSlider;

        [SerializeField] private VertexGradient textGradientEnabled;
        [SerializeField] private VertexGradient textGradientDisabled;

        private NiftiImage Brain { get; set; }

        private MeshFilter _brainMeshFilter;
        private Material _brainMaterial;
        private float _greyMatterThreshold = -1f;
        private bool _meshGenerated;  // whether or not the mesh has been created
        
        private static readonly int Color = Shader.PropertyToID("_Color");

        private void Start() {
            _brainMeshFilter = brain.GetComponent<MeshFilter>();
            _brainMaterial = brain.GetComponent<Renderer>().material;

            loadingPanel.SetActive(false);
            specPanel.SetActive(false);

            loadButton.interactable = true;
            loadButton.GetComponentInChildren<TextMeshProUGUI>().colorGradient = textGradientEnabled;

            segmentButton.interactable = false;
            segmentButton.GetComponentInChildren<TextMeshProUGUI>().colorGradient = textGradientDisabled;
            
            meshButton.interactable = false;
            meshButton.GetComponentInChildren<TextMeshProUGUI>().colorGradient = textGradientDisabled;
            
            quitButton.interactable = true;
            quitButton.GetComponentInChildren<TextMeshProUGUI>().colorGradient = textGradientEnabled;
        }

        private void Update() {
            if (Input.GetKeyDown(KeyCode.Space)) {
                specPanel.SetActive(!specPanel.activeInHierarchy);
            }
        }

        // load nifti image from disk (button event)
        public void LoadNiftiImage() {
            if (Brain == null) {
                // var path = Path.Combine(Directory.GetCurrentDirectory(), "Assets", "Resources", "brain.nii");
                var paths = StandaloneFileBrowser.OpenFilePanel("Select Nifti File", "./", "nii", false);
                if (paths.Length > 0) {
                    loadingPanel.SetActive(true);
                    StartCoroutine(LoadBrain(paths[0]));
                }
            }
        }

        // find segmentation threshold (button event)
        public void SegmentNiftiImage() {
            if (_greyMatterThreshold < 0) {
                loadingPanel.SetActive(true);
                StartCoroutine(SegmentBrain());
            }
        }

        // generate mesh using marching cubes (button event)
        public void SetMesh() {
            if (!_meshGenerated) {
                loadingPanel.SetActive(true);
                StartCoroutine(CreateBrainMesh());
            }
        }

        // change the main color alpha channel of the transparent shader (slider event)
        public void SetOpacity() {
            if (_meshGenerated) {
                var newColor = new Color(brainColor.r, brainColor.g, brainColor.b, alphaSlider.value);
                _brainMaterial.SetColor(Color, newColor);
            }
        }
        
        // quit the application
        public void QuitGame() => Application.Quit();

        // since our UI events are pretty heavy-duty tasks, it'd be better to wrap the computation
        // steps into coroutines so that we can fire them up as non-blocking background tasks.
        // the real OnClick event function should first switch on/off the UI elements properly, and
        // then start/schedule the coroutine, and return immediately, to do so, the coroutine must
        // yield in the first frame to let StartCoroutine() return, then start the computation in
        // the second frame, once finished, it will automatically update the UI elements.
        // the StartCoroutine() method won't return until it sees the first "yield return" statement.

        // without the use of coroutines, the event functions will not return until the computation is
        // complete, and the application window will freeze and not responding upon the button click.

        private IEnumerator LoadBrain(string path) {
            yield return null;  // temporarily return, then continue execution in the next frame

            // Brain = new NiftiImage(FileUtil.GetProjectRelativePath(path));
            Brain = new NiftiImage(path);
            
            segmentButton.interactable = true;
            segmentButton.GetComponentInChildren<TextMeshProUGUI>().colorGradient = textGradientEnabled;
            loadingPanel.SetActive(false);
        }
        
        private IEnumerator SegmentBrain() {
            yield return null;  // temporarily return, then continue execution in the next frame

            _greyMatterThreshold = OtsuThreshold.SegmentBrain(Brain.Data);
            Debug.Log($"The threshold intensity is {_greyMatterThreshold}");

            meshButton.interactable = true;
            meshButton.GetComponentInChildren<TextMeshProUGUI>().colorGradient = textGradientEnabled;
            loadingPanel.SetActive(false);
        }
        
        private IEnumerator CreateBrainMesh() {
            yield return null;  // temporarily return, then continue execution in the next frame

            var mesh = new Mesh();
            mesh.Clear();
            
            // must explicitly set the index format to 32 bit, which supports up to 4 billion vertices
            // the Unity default is 16 bit which only allows 65535 vertices (for performance issues)
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            
            MarchingCubes.GenerateMesh(Brain, _greyMatterThreshold,
                out var vertices, out var triangles, out var normals);
            
            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.normals = normals;
            mesh.RecalculateBounds();
            // mesh.RecalculateNormals();

            _brainMeshFilter.mesh = mesh;
            _meshGenerated = true;
            loadingPanel.SetActive(false);
        }
    }
}
