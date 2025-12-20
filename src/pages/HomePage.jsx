import React from 'react';
import Navbar from '../components/Navbar';
import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <div className="home-page">
      <Navbar />
      
      <main className="hero-section">
        <div className="hero-content">
          <h2 className="hero-title">SignLink</h2>
          <h1 className="hero-main-title">Real-Time Pakistan Sign Language Translator for Video Confrencing Platforms</h1>
          <p className="hero-description">
            Breaking communication barriers for the Deaf community with AI-powered sign language recognition 
            that works seamlessly with your favorite video conferencing tools.
          </p>
          <div className="hero-buttons">
            <Link to="/download" className="primary-button">Get Started</Link>
            <a href="#how-it-works" className="secondary-button">Learn More</a>
          </div>
        </div>
      </main>
    </div>
  );
};

export default HomePage;