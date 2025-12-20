import React from 'react';

const ExtensionCard = ({ browser, icon, onClick }) => {
  return (
    <div className="extension-card" onClick={onClick}>
      <div className="extension-icon">
        <img src={icon} alt={`${browser} icon`} />
      </div>
      <h3>{browser} Extension</h3>
      <button className="download-btn">Download</button>
    </div>
  );
};

export default ExtensionCard;