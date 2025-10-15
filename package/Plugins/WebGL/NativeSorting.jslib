// JavaScript helper for NativeSorting WASM plugin
// This provides initialization and worker management for the native sorting functionality

var NativeSortingPlugin = {
    
    // Initialize the plugin with WebGL context
    Initialize: function() {
        console.log("NativeSorting: JavaScript plugin initialized");
    },
    
    // Helper function to verify threading support
    CheckThreadingSupport: function() {
        var supportsThreads = typeof SharedArrayBuffer !== 'undefined';
        console.log("NativeSorting: Threading support available:", supportsThreads);
        return supportsThreads;
    }
};

// Auto-initialize when loaded
if (typeof Module !== 'undefined') {
    Module.onRuntimeInitialized = function() {
        NativeSortingPlugin.Initialize();
        CheckThreadingSupport();
    };
}
