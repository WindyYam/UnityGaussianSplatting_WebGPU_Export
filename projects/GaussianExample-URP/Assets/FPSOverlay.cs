using System;
using UnityEngine;
using UnityEngine.Scripting;

[Preserve]
static class FPSOverlayBootstrap
{
    // Creates the hidden GameObject automatically when the game starts
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Init()
    {
        var go = new GameObject("FPSOverlay_Runtime");
        go.hideFlags = HideFlags.HideAndDontSave;   // keep it out of hierarchy
        UnityEngine.Object.DontDestroyOnLoad(go);
        go.AddComponent<FPSOverlayBehaviour>();
    }
}

class FPSOverlayBehaviour : MonoBehaviour
{
    [SerializeField] float updateInterval = 1f;
    [SerializeField] Vector2 position = new Vector2(8, 8);
    [SerializeField] int fontSize = 14;
    [SerializeField] Color textColor = Color.green;

    float accum;
    int frames;
    float timeLeft;
    string lastText = "";
    GUIStyle style;

    void Awake()
    {
        timeLeft = updateInterval;
        style = new GUIStyle
        {
            alignment = TextAnchor.UpperLeft,
            fontSize = fontSize,
            richText = false
        };
        // ensure the text color is applied
        style.normal.textColor = textColor;
    }

    void Update()
    {
        float dt = Time.unscaledDeltaTime;
        timeLeft -= dt;
        accum += Math.Max(dt, 1e-6f);
        frames++;

        if (timeLeft <= 0f)
        {
            float fps = frames / accum;
            lastText = string.Format("{0:F1} FPS", fps);
            timeLeft = updateInterval;
            accum = 0f;
            frames = 0;
        }

        // Optional hotkey to toggle display
        if (Input.GetKeyDown(KeyCode.F11))
            enabled = !enabled;
    }

    void OnGUI()
    {
        if (!enabled) return;
        var rect = new Rect(position.x, position.y, 400, 64);
        GUI.Label(rect, lastText, style);
    }
}