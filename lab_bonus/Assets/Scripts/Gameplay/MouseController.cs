using UnityEngine;

namespace Gameplay {
    
    public class MouseController : MonoBehaviour {

        [SerializeField, Range(1f, 10f)]  private float rotationSpeed = 3f;

        private void OnMouseDrag() {
            var rotationVector = new Vector3(-Input.GetAxis("Mouse Y"), -Input.GetAxis("Mouse X"), 0);
            transform.Rotate(rotationVector * rotationSpeed * Time.deltaTime * 100f);
        }
    }
}