using UnityEngine;

public class CameraController : MonoBehaviour
{
    public Transform camera;
    [Header("Movement Settings")]
    public float moveSpeed = 1.5f;
    public float jumpSpeed = 3f;
    public float sprintSpeed = 4f;
    public float smoothTime = 0.1f;

    [Header("Mouse Look Settings")]
    public float mouseSensitivity = 2f;
    public float verticalClamp = 90f;
    public bool invertY = false;

    [Header("Key Bindings")]
    public KeyCode sprintKey = KeyCode.LeftShift;
    public KeyCode jumpKey = KeyCode.Space;
    public KeyCode crouchKey = KeyCode.LeftControl;

    [Header("Mobile Touch Settings")]
    public float moveJoystickSensitivity = 0.01f;   // smaller = stronger joystick
    public float lookSensitivity = 0.2f;           // swipe sensitivity

    [Header("Gyroscope Settings")]
    public bool enableGyro = true;
    public float gyroRotationMultiplier = 1f;

    private bool isGrounded = true;
    private bool isCrouching = false;

    // Capsule collider variables
    private CapsuleCollider capsuleCollider;
    private float originalColliderHeight;
    private Vector3 originalColliderCenter;

    // Mouse look variables
    private float mouseX;
    private float mouseY;

    // Gyro state
    private bool gyroInitialized = false;
    // Track last gyro target so we only apply rotation when the gyro value actually changes.
    private Quaternion lastGyroTarget = Quaternion.identity;
    private bool hasLastGyroTarget = false;
    // Base yaw mapping so gyro yaw is applied as a delta to the transform's initial yaw
    private float gyroBaseDeviceYaw = 0f;
    private float gyroBaseTransformYaw = 0f;
    private bool gyroBaseSet = false;

    // --- Touch joystick state ---
    private Vector2 moveAxis;   // virtual WASD
    private Vector2 lookAxis;   // virtual mouse delta
    private int leftFingerId = -1;
    private int rightFingerId = -1;
    private Vector2 leftTouchStart;
    private Vector2 rightTouchStart;

    void Start()
    {
        // Lock cursor to center of screen
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;

        // Get and store capsule collider properties
        capsuleCollider = GetComponent<CapsuleCollider>();
        if (capsuleCollider != null)
        {
            originalColliderHeight = capsuleCollider.height;
            originalColliderCenter = capsuleCollider.center;
        }

        // Initialize gyroscope when available and requested
        if (enableGyro && SystemInfo.supportsGyroscope)
        {
            Input.gyro.enabled = true;
            gyroInitialized = true;
        }
    }

    void Update()
    {
        // Read gyro first to drive rotation on mobile/web when available
        if (enableGyro && gyroInitialized)
        {
            HandleGyro();
        }

        HandleTouchInput();   // NEW

        HandleMouseLook();
        HandleMovement();
        HandleSpecialMovement();

        // Toggle cursor lock with Escape key
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            ToggleCursorLock();
        }
    }

    // Use device gyroscope to orient the player/camera so that the camera points where the phone's back is pointing.
    void HandleGyro()
    {
        // Read attitude
        Quaternion q = Input.gyro.attitude;

        // Convert gyro quaternion to Unity's left-handed coordinate system and adjust axes.
        Quaternion deviceRotation = new Quaternion(q.x, q.y, -q.z, -q.w);
        // Only apply rotation when the gyro-reported target changes.
        // This avoids repeatedly applying identical rotations (useful to detect devices without a real gyro).
        if (!hasLastGyroTarget || !Quaternion.Equals(deviceRotation, lastGyroTarget))
        {
            lastGyroTarget = deviceRotation;
            hasLastGyroTarget = true;

            // Align device axes: portrait mode (screen up), phone back = camera forward
            Quaternion alignedRotation = Quaternion.Euler(90f, 0f, 0f) * deviceRotation;

            // Decompose aligned rotation into yaw (y) and pitch/roll (x/z).
            Vector3 e = alignedRotation.eulerAngles;

            float yaw = e.y;
            float pitch = e.x;
            float roll = e.z;

            // Convert to -180..180 range for sensible clamping/behavior
            if (yaw > 180f) yaw -= 360f;
            if (pitch > 180f) pitch -= 360f;
            if (roll > 180f) roll -= 360f;

            // Clamp pitch to the configured vertical clamp (so device can't flip camera upside down)
            pitch = Mathf.Clamp(pitch, -verticalClamp, verticalClamp);

            // Establish base yaw mapping on first valid gyro reading so we add increments to the transform's initial yaw
            if (!gyroBaseSet)
            {
                gyroBaseDeviceYaw = yaw;
                gyroBaseTransformYaw = transform.eulerAngles.y;
                if (gyroBaseTransformYaw > 180f) gyroBaseTransformYaw -= 360f;
                gyroBaseSet = true;
            }

            // Compute yaw delta relative to base, normalize to -180..180
            float yawDelta = yaw - gyroBaseDeviceYaw;
            if (yawDelta > 180f) yawDelta -= 360f;
            if (yawDelta < -180f) yawDelta += 360f;

            float targetYaw = gyroBaseTransformYaw + yawDelta;

            // Apply new rotations: yaw as delta on the transform, pitch/roll to the local camera
            transform.rotation = Quaternion.Euler(0f, targetYaw, 0f);
            camera.localRotation = Quaternion.Euler(pitch, 0f, roll);
        }
    }

    void HandleMouseLook()
    {
        // If there are active touches, don't read the desktop mouse axes because many browsers
        // synthesize mouse events from touch. Only use mouse input when no touches are present.
        float mx = 0f;
        float my = 0f;

        if (Input.touchCount == 0)
        {
            // desktop mouse input
            mx = Input.GetAxis("Mouse X") * mouseSensitivity;
            my = Input.GetAxis("Mouse Y") * mouseSensitivity;
        }

        // Add mobile look input
        mx += lookAxis.x;
        my += lookAxis.y;

        if (invertY) my = -my;

        // Horizontal rotation
        transform.Rotate(0, mx, 0, Space.World);

        // Directly rotate camera's localRotation pitch with clamping, keeping original roll
        Vector3 currentEuler = camera.localRotation.eulerAngles;
        float pitch = currentEuler.x;
        float roll = currentEuler.z;

        // Convert pitch to -180..180 range
        if (pitch > 180f) pitch -= 360f;

        pitch -= my;
        pitch = Mathf.Clamp(pitch, -verticalClamp, verticalClamp);

        camera.localRotation = Quaternion.Euler(pitch, 0, roll);
    }

    void HandleMovement()
    {
        // Get desktop input
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");

        // Add mobile joystick input
        horizontal += moveAxis.x;
        vertical += moveAxis.y;

        // Determine current speed
        float currentSpeed = Input.GetKey(sprintKey) ? sprintSpeed : moveSpeed;

        // Directions
        Vector3 forward = transform.forward;
        forward.y = 0;
        forward.Normalize();

        Vector3 right = transform.right;
        right.y = 0;
        right.Normalize();

        // Move
        Vector3 direction = right * horizontal + forward * vertical;
        direction = Vector3.ClampMagnitude(direction, 1f);
        transform.Translate(direction * currentSpeed * Time.deltaTime, Space.World);
    }

    void HandleSpecialMovement()
    {
        if (Input.GetKeyDown(jumpKey) && isGrounded)
        {
            Vector3 jumpSpeed = Vector3.up * this.jumpSpeed * 2f;
            GetComponent<Rigidbody>().linearVelocity += jumpSpeed;
        }

        if (Input.GetKeyDown(crouchKey))
        {
            ToggleCrouch();
        }
    }

    void ToggleCrouch()
    {
        isCrouching = !isCrouching;

        if (isCrouching)
        {
            if (capsuleCollider != null)
            {
                float newHeight = originalColliderHeight * 0.5f;
                float heightDifference = originalColliderHeight - newHeight;

                capsuleCollider.height = newHeight;
                capsuleCollider.center = new Vector3(
                    originalColliderCenter.x,
                    originalColliderCenter.y + (heightDifference * 0.5f),
                    originalColliderCenter.z
                );
            }
        }
        else
        {
            if (capsuleCollider != null)
            {
                capsuleCollider.height = originalColliderHeight;
                capsuleCollider.center = originalColliderCenter;
            }
        }
    }

    void ToggleCursorLock()
    {
        if (Cursor.lockState == CursorLockMode.Locked)
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }
        else
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
    }

    // --- Mobile touch joystick handler ---
    void HandleTouchInput()
    {
        moveAxis = Vector2.zero;
        lookAxis = Vector2.zero;

        bool leftFound = false;
        bool rightFound = false;

        for (int i = 0; i < Input.touchCount; i++)
        {
            Touch t = Input.GetTouch(i);

            // Assign fingers only on Began using the initial touch position (so assignment sticks
            // even if the finger moves across the screen center).
            if (t.phase == TouchPhase.Began)
            {
                // When gyro is active (mobile web), treat the first touch as movement regardless of side
                bool beganLeft = t.position.x < Screen.width / 2f;
                if (beganLeft && leftFingerId == -1)
                {
                    leftFingerId = t.fingerId;
                    leftTouchStart = t.position;
                    leftFound = true;
                    continue;
                }
                else if (!beganLeft && rightFingerId == -1)
                {
                    rightFingerId = t.fingerId;
                    rightTouchStart = t.position;
                    rightFound = true;
                    continue;
                }
            }
            else
            {
                Debug.Log($"t.fingerId: {t.fingerId}, leftFingerId: {leftFingerId}, rightFingerId: {rightFingerId}");
                // Process movement for the finger previously assigned to the left joystick
                if (t.fingerId == leftFingerId)
                {
                    if (t.phase == TouchPhase.Moved || t.phase == TouchPhase.Stationary)
                    {
                        // Use the delta from the initial touch start to behave like a joystick.
                        Vector2 delta = t.position - leftTouchStart;

                        // Normalize with a reasonable radius so full deflection ~1.
                        float maxRadius = Mathf.Max(Screen.width, Screen.height) * 0.25f;
                        moveAxis = Vector2.ClampMagnitude(delta / maxRadius, 1f);

                        // Optionally scale sensitivity (keep existing variable semantics)
                        moveAxis *= (1f / Mathf.Max(moveJoystickSensitivity, 0.0001f));

                        leftFound = true;
                    }

                    if (t.phase == TouchPhase.Ended || t.phase == TouchPhase.Canceled)
                    {
                        leftFingerId = -1;
                    }
                }
                else if (t.fingerId == rightFingerId)
                {
                    if (t.phase == TouchPhase.Moved || t.phase == TouchPhase.Stationary)
                    {
                        // Use deltaPosition for smoother look deltas
                        lookAxis = t.deltaPosition * lookSensitivity;
                        rightFound = true;
                    }

                    if (t.phase == TouchPhase.Ended || t.phase == TouchPhase.Canceled)
                    {
                        rightFingerId = -1;
                    }
                }
            }
        }

        // If assigned finger ids were not found in the current touches, clear them to avoid stale ids.
        if (!leftFound && leftFingerId != -1)
            leftFingerId = -1;
        if (!rightFound && rightFingerId != -1)
            rightFingerId = -1;
    }
}
