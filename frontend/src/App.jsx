import React, { useState, useRef, useEffect } from 'react'

function App() {
  const [inputFile, setInputFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [outputUrl, setOutputUrl] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [videos, setVideos] = useState([]);
  const [colabUrl, setColabUrl] = useState('https://7eef9ca5e047724e-136-115-156-7.serveousercontent.com');
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [isLoadingVideos, setIsLoadingVideos] = useState(false);
  const [downloadingVideo, setDownloadingVideo] = useState(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionStatus, setDetectionStatus] = useState('');
  const [gunDetected, setGunDetected] = useState(false);
  const [currentFPS, setCurrentFPS] = useState(0);
  const [detectedImages, setDetectedImages] = useState([]);
  const [showDetectedImages, setShowDetectedImages] = useState(false);
  const [framesSent, setFramesSent] = useState(0);
  const [backendStatus, setBackendStatus] = useState('Unknown');
  
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);
  const intervalRef = useRef(null);
  const sessionIdRef = useRef(Date.now().toString());
  const frameIndexRef = useRef(0);
  
  // ✅ FIX 1: Use ref for active state to avoid closure issues
  const isActiveRef = useRef(false);
  
  // ✅ FIX 2: Use ref for sending state to prevent concurrent sends
  const isSendingRef = useRef(false);

  // Test backend connection
  const testBackend = async () => {
    console.log("Testing backend...");
    try {
      const response = await fetch(`${colabUrl}/health`);
      if (response.ok) {
        const data = await response.json();
        setBackendStatus(`Connected (${data.models_loaded} models)`);
        setErrorMsg(`✅ Backend connected!`);
        setTimeout(() => setErrorMsg(''), 3000);
        return true;
      }
    } catch (error) {
      setBackendStatus(`Disconnected`);
      setErrorMsg(`❌ Cannot connect: ${error.message}`);
      return false;
    }
  };

  // Send frame to backend
  const sendFrame = async () => {
    // ✅ Check ref instead of state to avoid stale closure
    if (!isActiveRef.current) {
      console.log("❌ Webcam not active (ref), skipping");
      return;
    }
    
    if (isSendingRef.current) {
      console.log("⏳ Already sending, skipping");
      return;
    }
    
    if (!videoRef.current) {
      console.log("❌ No video ref");
      return;
    }
    
    if (!videoRef.current.videoWidth || !videoRef.current.videoHeight) {
      console.log("⏳ Video dimensions not ready yet");
      return;
    }
    
    isSendingRef.current = true;
    const currentFrameIndex = frameIndexRef.current;
    console.log(`📸 Capturing frame ${currentFrameIndex}...`);
    
    try {
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      const frameBlob = await new Promise((resolve) => {
        canvas.toBlob(resolve, 'image/jpeg', 0.7);
      });
      
      console.log(`📦 Frame ${currentFrameIndex}: ${frameBlob.size} bytes, ${canvas.width}x${canvas.height}`);
      
      setIsDetecting(true);
      setDetectionStatus(`Analyzing frame ${currentFrameIndex}...`);
      
      const formData = new FormData();
      formData.append('frame', frameBlob, `frame_${currentFrameIndex}.jpg`);
      formData.append('session_id', sessionIdRef.current);
      formData.append('frame_index', currentFrameIndex.toString());
      
      const response = await fetch(`${colabUrl}/detect-single`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const result = await response.json();
      console.log(`✅ Frame ${currentFrameIndex} result: gun_detected=${result.gun_detected}`);
      
      // Update counters
      setFramesSent(prev => prev + 1);
      frameIndexRef.current++;
      
      if (result.gun_detected) {
        console.log(`🚨 GUN DETECTED in frame ${currentFrameIndex}!`);
        setGunDetected(true);
        setDetectionStatus(`⚠️ GUN DETECTED in frame ${currentFrameIndex}!`);
        setErrorMsg(`⚠️ GUN DETECTED!`);
        setTimeout(() => setErrorMsg(''), 5000);
        
        const imageUrl = URL.createObjectURL(frameBlob);
        setDetectedImages(prev => [{
          id: Date.now(),
          url: imageUrl,
          timestamp: new Date().toLocaleTimeString(),
          frameIndex: currentFrameIndex
        }, ...prev]);
        setShowDetectedImages(true);
      } else {
        setGunDetected(false);
        setDetectionStatus(`Monitoring - Frame ${currentFrameIndex} (no gun)`);
      }
    } catch (error) {
      console.error(`❌ Frame error:`, error);
      setDetectionStatus(`Error: ${error.message}`);
    } finally {
      setIsDetecting(false);
      isSendingRef.current = false;
    }
  };

  // Start sending frames at 1 FPS
  const startFrameCapture = () => {
    console.log("🚀 Starting frame capture at 1 frame per second");
    console.log(`isActiveRef.current = ${isActiveRef.current}`);
    
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    // Send first frame after 1 second
    setTimeout(() => {
      console.log("⏰ Sending first frame...");
      sendFrame();
    }, 1000);
    
    // Then send every 1 second
    intervalRef.current = setInterval(() => {
      console.log("⏰ Interval tick - sending frame...");
      sendFrame();
    }, 1000);
  };

  const stopWebcamAndDetection = () => {
    console.log("Stopping webcam...");
    
    // ✅ Clear the interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    // ✅ Set ref to false immediately
    isActiveRef.current = false;
    
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    
    setIsWebcamActive(false);
    setIsDetecting(false);
    setGunDetected(false);
    setDetectionStatus('');
    setCurrentFPS(0);
    
    // End session on backend
    if (sessionIdRef.current && colabUrl) {
      fetch(`${colabUrl}/end-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionIdRef.current })
      }).catch(console.error);
    }
    
    console.log(`Webcam stopped. Total frames sent: ${framesSent}`);
  };

  const startWebcam = async () => {
    console.log("========================================");
    console.log("STARTING WEBCAM");
    console.log("========================================");
    
    if (isWebcamActive) {
      stopWebcamAndDetection();
      return;
    }
    
    // Reset state
    setGunDetected(false);
    setDetectionStatus('Starting webcam...');
    setFramesSent(0);
    frameIndexRef.current = 0;
    sessionIdRef.current = Date.now().toString();
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        },
        audio: false 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        videoRef.current.onloadedmetadata = () => {
          console.log(`✅ Video ready: ${videoRef.current.videoWidth}x${videoRef.current.videoHeight}`);
          videoRef.current.play();
          
          // ✅ Set both state AND ref
          setIsWebcamActive(true);
          isActiveRef.current = true;
          
          setDetectionStatus('Webcam active - Starting detection...');
          
          // Start sending frames
          startFrameCapture();
        };
      }
    } catch (error) {
      console.error('Webcam error:', error);
      if (error.name === 'NotAllowedError') {
        setErrorMsg('Please allow camera access');
      } else if (error.name === 'NotFoundError') {
        setErrorMsg('No camera found');
      } else {
        setErrorMsg(`Webcam error: ${error.message}`);
      }
      setIsWebcamActive(false);
      isActiveRef.current = false;
    }
  };

  const fetchVideos = async () => {
    if (!colabUrl) return;
    
    setIsLoadingVideos(true);
    try {
      const response = await fetch(`${colabUrl}/list-videos`);
      if (response.ok) {
        const data = await response.json();
        setVideos(data.videos || []);
      }
    } catch (error) {
      console.error('Error fetching videos:', error);
    } finally {
      setIsLoadingVideos(false);
    }
  };

  const downloadVideo = async (filename, url) => {
    setDownloadingVideo(filename);
    try {
      const response = await fetch(`${colabUrl}${url}`);
      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(downloadUrl);
      setErrorMsg(`✅ Downloaded: ${filename}`);
      setTimeout(() => setErrorMsg(''), 3000);
    } catch (error) {
      setErrorMsg(`Failed to download: ${error.message}`);
    } finally {
      setDownloadingVideo(null);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.type.startsWith('video/')) {
      setErrorMsg('Please select a valid video file');
      return;
    }
    setInputFile(file);
    setOutputUrl(null);
  };

  const clearMedia = () => {
    setInputFile(null);
    setOutputUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleProcess = async () => {
    if (!inputFile) {
      setErrorMsg('Please select a file first');
      return;
    }
    
    setIsProcessing(true);
    try {
      const formData = new FormData();
      formData.append('file', inputFile);
      const response = await fetch(`${colabUrl}/process`, {
        method: 'POST',
        body: formData,
      });
      const blob = await response.blob();
      setOutputUrl(URL.createObjectURL(blob));
      setTimeout(fetchVideos, 2000);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!outputUrl) return;
    const link = document.createElement('a');
    link.href = outputUrl;
    link.download = `detected_video_${Date.now()}.mp4`;
    link.click();
  };

  // Calculate FPS for display
  useEffect(() => {
    if (!isWebcamActive) return;
    let lastTime = performance.now();
    const calculateFPS = () => {
      if (!isWebcamActive) return;
      const now = performance.now();
      const delta = now - lastTime;
      const fps = Math.round(1000 / delta);
      setCurrentFPS(fps);
      lastTime = now;
      requestAnimationFrame(calculateFPS);
    };
    const rafId = requestAnimationFrame(calculateFPS);
    return () => cancelAnimationFrame(rafId);
  }, [isWebcamActive]);

  // ✅ FIX 3: Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Initial fetch
  useEffect(() => {
    if (colabUrl) {
      testBackend();
      fetchVideos();
      const interval = setInterval(fetchVideos, 30000);
      return () => clearInterval(interval);
    }
  }, [colabUrl]);

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)', padding: '20px' }}>
      <style>{`
        .spinner {
          width: 24px;
          height: 24px;
          border: 3px solid rgba(255,255,255,0.3);
          border-top-color: white;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        .detected-image:hover {
          transform: scale(1.05);
          transition: transform 0.3s;
        }
      `}</style>
      
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '40px' }}>
          <h1 style={{ 
            fontSize: '2.5rem', 
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: '10px'
          }}>
            Gun Detection & Tracking System
          </h1>
          <p style={{ color: '#a0a0a0' }}>
            Upload video • AI detects guns and tracks holders • Get annotated results
          </p>
        </div>

        {/* Colab URL Configuration */}
        <div style={{ 
          background: 'rgba(0,0,0,0.5)', 
          borderRadius: '12px', 
          padding: '12px 20px', 
          marginBottom: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px'
        }}>
          <div style={{ flex: 1 }}>
            <span style={{ color: '#888', fontSize: '12px' }}>Colab Backend URL:</span>
            {showUrlInput ? (
              <input
                type="text"
                value={colabUrl}
                onChange={(e) => setColabUrl(e.target.value)}
                style={{
                  marginLeft: '8px',
                  padding: '4px 8px',
                  background: '#2d3748',
                  border: '1px solid #4a5568',
                  borderRadius: '4px',
                  color: 'white',
                  width: '300px'
                }}
              />
            ) : (
              <span style={{ color: '#a78bfa', marginLeft: '8px', fontSize: '12px', wordBreak: 'break-all' }}>{colabUrl}</span>
            )}
          </div>
          <div>
            <button
              onClick={() => setShowUrlInput(!showUrlInput)}
              style={{
                background: '#4a5568',
                color: 'white',
                border: 'none',
                padding: '4px 12px',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px',
                marginRight: '8px'
              }}
            >
              {showUrlInput ? 'Save URL' : 'Edit URL'}
            </button>
            <button
              onClick={testBackend}
              style={{
                background: '#10b981',
                color: 'white',
                border: 'none',
                padding: '4px 12px',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px',
                marginRight: '8px'
              }}
            >
              🔌 Test Backend
            </button>
            <button
              onClick={fetchVideos}
              style={{
                background: '#3b82f6',
                color: 'white',
                border: 'none',
                padding: '4px 12px',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              🔄 Refresh Videos
            </button>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
          {/* Input Card */}
          <div style={{ 
            background: 'rgba(255,255,255,0.1)', 
            backdropFilter: 'blur(10px)',
            borderRadius: '16px',
            padding: '24px',
            border: '1px solid rgba(255,255,255,0.2)'
          }}>
            <h2 style={{ color: 'white', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>📤</span> Upload & Process
            </h2>
            
            <button
              onClick={startWebcam}
              style={{
                width: '100%',
                marginBottom: '16px',
                padding: '12px',
                background: isWebcamActive ? '#ef4444' : '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '14px',
                fontWeight: 'bold',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                transition: 'all 0.3s'
              }}
            >
              <span>🎥</span>
              <span>{isWebcamActive ? 'Stop Webcam' : 'Connect Webcam'}</span>
            </button>

            {/* Frames Sent Counter */}
            {isWebcamActive && (
              <div style={{
                background: 'rgba(0,0,0,0.5)',
                borderRadius: '8px',
                padding: '15px',
                marginBottom: '15px',
                textAlign: 'center'
              }}>
                <div style={{ color: '#a78bfa', fontSize: '36px', fontWeight: 'bold' }}>{framesSent}</div>
                <div style={{ color: '#888', fontSize: '12px' }}>Frames Sent to Backend</div>
                <div style={{ color: '#4ade80', fontSize: '11px', marginTop: '5px' }}>📸 1 frame/second</div>
              </div>
            )}

            {/* Live Webcam Feed */}
            <div style={{ 
              background: '#000',
              borderRadius: '12px',
              overflow: 'hidden',
              border: `2px solid ${gunDetected ? '#ef4444' : (isWebcamActive ? '#10b981' : '#4a5568')}`,
              minHeight: '300px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative'
            }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{
                  width: '100%',
                  height: 'auto',
                  maxHeight: '300px',
                  objectFit: 'contain'
                }}
              />
              {!isWebcamActive && (
                <div style={{
                  position: 'absolute',
                  color: '#888',
                  textAlign: 'center',
                  padding: '20px',
                  pointerEvents: 'none'
                }}>
                  <div style={{ fontSize: '48px', marginBottom: '10px' }}>📷</div>
                  <p>Click "Connect Webcam" to start</p>
                </div>
              )}
              {isWebcamActive && (
                <>
                  <div style={{
                    position: 'absolute',
                    top: '10px',
                    left: '10px',
                    background: 'rgba(0,0,0,0.7)',
                    color: gunDetected ? '#ef4444' : '#10b981',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontSize: '12px',
                    fontWeight: 'bold',
                    pointerEvents: 'none'
                  }}>
                    {gunDetected ? '🔴 GUN DETECTED' : '🟢 MONITORING'}
                  </div>
                  <div style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    background: 'rgba(0,0,0,0.7)',
                    color: '#fff',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    pointerEvents: 'none'
                  }}>
                    Camera FPS: {currentFPS} | 📸 1 fps to AI
                  </div>
                  {detectionStatus && (
                    <div style={{
                      position: 'absolute',
                      bottom: '10px',
                      left: '10px',
                      right: '10px',
                      background: 'rgba(0,0,0,0.8)',
                      color: gunDetected ? '#ef4444' : '#a78bfa',
                      padding: '6px 12px',
                      borderRadius: '4px',
                      fontSize: '11px',
                      textAlign: 'center',
                      pointerEvents: 'none'
                    }}>
                      {detectionStatus}
                    </div>
                  )}
                  {isDetecting && (
                    <div style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      background: 'rgba(0,0,0,0.8)',
                      padding: '8px 16px',
                      borderRadius: '8px',
                      fontSize: '12px',
                      color: 'white',
                      pointerEvents: 'none'
                    }}>
                      <div className="spinner" style={{ display: 'inline-block', marginRight: '8px' }}></div>
                      Analyzing...
                    </div>
                  )}
                </>
              )}
            </div>
            
            <div 
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${inputFile ? '#4ade80' : 'rgba(255,255,255,0.3)'}`,
                borderRadius: '12px',
                padding: '40px',
                textAlign: 'center',
                cursor: 'pointer',
                background: 'rgba(0,0,0,0.3)',
                transition: 'all 0.3s',
                marginTop: '20px'
              }}
            >
              <input 
                ref={fileInputRef}
                type="file" 
                accept="video/*" 
                style={{ display: 'none' }}
                onChange={handleFileSelect}
              />
              <div style={{ fontSize: '48px', marginBottom: '12px' }}>🎥</div>
              <p style={{ color: 'white' }}>
                {inputFile ? inputFile.name : 'Click to select video'}
              </p>
              <p style={{ color: '#888', fontSize: '12px', marginTop: '8px' }}>
                MP4, MOV, AVI (Max 500MB)
              </p>
            </div>

            {inputFile && (
              <div style={{ marginTop: '20px' }}>
                <div style={{ 
                  background: 'rgba(0,0,0,0.3)', 
                  borderRadius: '12px', 
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <video 
                    src={URL.createObjectURL(inputFile)} 
                    controls 
                    style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '8px' }}
                    onLoadedMetadata={(e) => URL.revokeObjectURL(e.target.src)}
                  />
                  <button
                    onClick={clearMedia}
                    style={{
                      marginTop: '12px',
                      background: '#ef4444',
                      color: 'white',
                      border: 'none',
                      padding: '6px 12px',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '12px'
                    }}
                  >
                    Remove Video
                  </button>
                </div>
                
                <button
                  onClick={handleProcess}
                  disabled={isProcessing}
                  style={{
                    width: '100%',
                    marginTop: '20px',
                    padding: '12px',
                    background: isProcessing ? '#4b5563' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    fontSize: '16px',
                    fontWeight: 'bold',
                    cursor: isProcessing ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '8px'
                  }}
                >
                  {isProcessing ? (
                    <>
                      <div className="spinner"></div>
                      <span>Processing on Colab...</span>
                    </>
                  ) : (
                    <>
                      <span>🚀</span>
                      <span>Detect Guns with Colab</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Output & Videos Card */}
          <div style={{ 
            background: 'rgba(255,255,255,0.1)', 
            backdropFilter: 'blur(10px)',
            borderRadius: '16px',
            padding: '24px',
            border: '1px solid rgba(255,255,255,0.2)'
          }}>
            <h2 style={{ color: 'white', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>✨</span> Processed Videos
            </h2>
            
            {isProcessing ? (
              <div style={{
                textAlign: 'center',
                padding: '40px 20px',
                background: 'rgba(0,0,0,0.3)',
                borderRadius: '12px'
              }}>
                <div className="spinner" style={{ margin: '0 auto 16px', width: '40px', height: '40px' }}></div>
                <p style={{ color: '#a78bfa' }}>Processing video...</p>
              </div>
            ) : outputUrl ? (
              <div style={{ marginBottom: '24px' }}>
                <div style={{
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: '12px',
                  padding: '16px',
                  textAlign: 'center'
                }}>
                  <video 
                    src={outputUrl} 
                    controls 
                    style={{ maxWidth: '100%', maxHeight: '250px', borderRadius: '8px' }}
                  />
                  <button
                    onClick={handleDownload}
                    style={{
                      width: '100%',
                      marginTop: '12px',
                      padding: '10px',
                      background: '#4ade80',
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      fontSize: '14px',
                      fontWeight: 'bold',
                      cursor: 'pointer'
                    }}
                  >
                    ⬇️ Download Latest Processed Video
                  </button>
                </div>
              </div>
            ) : null}

            {/* Archived Videos List */}
            <div>
              <h3 style={{ color: 'white', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span>📋</span> Archived Videos
                {isLoadingVideos && <div className="spinner" style={{ width: '16px', height: '16px' }}></div>}
              </h3>
              
              {videos.length === 0 && !isLoadingVideos && (
                <div style={{
                  textAlign: 'center',
                  padding: '40px',
                  background: 'rgba(0,0,0,0.3)',
                  borderRadius: '12px',
                  color: '#888'
                }}>
                  No processed videos yet. Upload and process a video to get started.
                </div>
              )}
              
              <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                {videos.map((video) => (
                  <div 
                    key={video.filename}
                    style={{
                      background: 'rgba(0,0,0,0.3)',
                      borderRadius: '8px',
                      padding: '12px',
                      marginBottom: '8px'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ color: 'white', fontSize: '13px', fontWeight: 'bold', marginBottom: '4px', wordBreak: 'break-all' }}>
                          {video.filename}
                        </div>
                        <div style={{ color: '#888', fontSize: '11px' }}>
                          {formatFileSize(video.size)} • {formatDate(video.created)}
                        </div>
                      </div>
                      <button
                        onClick={() => downloadVideo(video.filename, video.url)}
                        disabled={downloadingVideo === video.filename}
                        style={{
                          background: downloadingVideo === video.filename ? '#4b5563' : '#3b82f6',
                          color: 'white',
                          border: 'none',
                          padding: '6px 12px',
                          borderRadius: '6px',
                          fontSize: '12px',
                          cursor: downloadingVideo === video.filename ? 'not-allowed' : 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px'
                        }}
                      >
                        {downloadingVideo === video.filename ? (
                          <>
                            <div className="spinner" style={{ width: '12px', height: '12px' }}></div>
                            <span>Downloading...</span>
                          </>
                        ) : (
                          <>
                            <span>⬇️</span>
                            <span>Download</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Detected Images Section */}
        {showDetectedImages && detectedImages.length > 0 && (
          <div style={{ 
            marginTop: '24px',
            background: 'rgba(0,0,0,0.5)',
            borderRadius: '16px',
            padding: '20px',
            border: '1px solid rgba(239,68,68,0.3)'
          }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              marginBottom: '16px'
            }}>
              <h3 style={{ color: '#ef4444', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span>⚠️</span> Gun Detected Images ({detectedImages.length})
              </h3>
              <button
                onClick={() => setShowDetectedImages(false)}
                style={{
                  background: '#4a5568',
                  color: 'white',
                  border: 'none',
                  padding: '4px 12px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
              >
                Hide
              </button>
            </div>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
              gap: '16px',
              maxHeight: '500px',
              overflowY: 'auto',
              padding: '8px'
            }}>
              {detectedImages.map((img) => (
                <div key={img.id} className="detected-image" style={{
                  background: 'rgba(0,0,0,0.6)',
                  borderRadius: '12px',
                  padding: '12px',
                  border: '2px solid #ef4444'
                }}>
                  <img 
                    src={img.url} 
                    alt={`Gun detected at ${img.timestamp}`}
                    style={{
                      width: '100%',
                      height: 'auto',
                      borderRadius: '8px',
                      cursor: 'pointer'
                    }}
                    onClick={() => window.open(img.url, '_blank')}
                  />
                  <div style={{
                    marginTop: '8px',
                    fontSize: '11px',
                    color: '#ef4444',
                    textAlign: 'center'
                  }}>
                    🚨 Gun Detected at {img.timestamp} (Frame {img.frameIndex})
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {errorMsg && (
          <div style={{
            marginTop: '20px',
            padding: '12px',
            background: errorMsg.includes('✅') ? 'rgba(74,222,128,0.2)' : (errorMsg.includes('GUN') ? 'rgba(239,68,68,0.3)' : 'rgba(239,68,68,0.2)'),
            border: `1px solid ${errorMsg.includes('✅') ? '#4ade80' : (errorMsg.includes('GUN') ? '#ef4444' : '#ef4444')}`,
            borderRadius: '8px',
            color: errorMsg.includes('✅') ? '#4ade80' : '#fecaca',
            fontSize: '14px',
            textAlign: 'center'
          }}>
            {errorMsg}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;