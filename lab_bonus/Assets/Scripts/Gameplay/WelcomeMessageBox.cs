using UnityEngine;

namespace Gameplay {
    
    public class WelcomeMessageBox : MonoBehaviour {
        [SerializeField] private string title = default;
        [SerializeField, Multiline] private string message = default;
        [SerializeField] private string buttonText = "Confirm";

        private bool _showWindow = true;
        private const float Margin = 500f;  // leave a minimum of 500 pixels space from the screen border

        private void OnGUI() {
            if (!_showWindow) { return; }
            
            GUI.skin.window.fontSize = 18;  // title font size
            GUI.skin.label.fontSize = 14;   // message font size
            GUI.skin.button.fontSize = 14;  // button font size

            GUI.skin.window.alignment = TextAnchor.UpperCenter;
            GUI.skin.window.padding = new RectOffset(10, 10, 40, 0);
            GUI.skin.label.padding = new RectOffset(2, 2, 2, 2);

            var boxWidth = 550f;
            var size = new Vector2(boxWidth, GUI.skin.label.CalcHeight(new GUIContent(message), boxWidth));
            var maxWidth = Mathf.Min(Screen.width - Margin, size.x);
            var left = Screen.width * 0.5f - maxWidth * 0.5f;
            var top = Screen.height * 0.4f - size.y * 0.5f;

            var windowRect = new Rect(left, top, maxWidth, size.y);
            GUI.contentColor = Color.red;
            GUILayout.Window(123, windowRect, id => DrawWindow(id, maxWidth), title);
        }

        private void DrawWindow(int id, float maxWidth) {
            GUI.contentColor = Color.white;
            GUILayout.Space(15);  // space between the title and the message box
            
            GUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Label(message);
            GUILayout.EndVertical();
            
            GUILayout.Space(6);  // space between the message box and the button
            
            if (GUILayout.Button(buttonText)) {
                _showWindow = false;
            }
            
            GUILayout.Space(10);  // space between the button and the window border
        }
    }
}