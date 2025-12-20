import React from 'react';
import Navbar from '../components/Navbar';
import ExtensionCard from '../components/ExtensionCard';

const DownloadPage = () => {
  const handleChromeDownload = () => {
    // In a real app, this would link to Chrome Web Store
    console.log('Downloading Chrome extension...');
  };

  const handleEdgeDownload = () => {
    // In a real app, this would link to Edge Add-ons store
    console.log('Downloading Edge extension...');
  };

  return (
    <div className="download-page">
      <Navbar />
      
      <main className="download-section">
        <div className="download-content">
          <h2 className="download-title">SignLink</h2>
          <h1 className="download-main-title">Download SignLink</h1>
          <p className="download-description">
            Get started with real-time sign language translation in just a few clicks. 
            Available for Chrome and Edge browsers.
          </p>
          
          <div className="extensions-container">
            <div className="extension-wrapper" onClick={handleChromeDownload}>
              <div className="extension-card">
                <div className="extension-icon">
                  <div className="chrome-icon">C</div>
                </div>
                <h3>Chrome Extension</h3>
                <button className="download-btn">Download</button>
              </div>
            </div>
            
            <div className="extension-wrapper" onClick={handleEdgeDownload}>
              <div className="extension-card">
                <div className="extension-icon">
                  <div className="edge-icon">E</div>
                </div>
                <h3>Edge Extension</h3>
                <button className="download-btn">Download</button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default DownloadPage;