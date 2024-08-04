import React, { useState } from 'react';
import './Home.css';
import Navbar from './Components/Navbar';
import SidebarLeft from './Area/SidebarLeft';
import SidebarRight from './Area/SidebarRight';
import Footer from './Area/Footer';
import MapSwap from './Area/MapSwap';
import Content from './Area/Content';
import ProVideoSwap from './Area/ProVideoSwap';
import ClipSwap from './Area/ClipSwap';

function Home() {
  const [selectedPersonData, setSelectedPersonData] = useState([]);
  const [selectedVideoUrl, setSelectedVideoUrl] = useState('');
  const [selectedCamData, setSelectedCamData] = useState([]);
  const [showClipSwap, setShowClipSwap] = useState(false);
  const [showMapSwap, setShowMapSwap] = useState(false);
  const [showProVideoSwap, setShowProVideoSwap] = useState(false);

  const handlePersonSelect = (personData, videoUrl) => {
    setSelectedPersonData(personData);
    setSelectedVideoUrl(videoUrl);
    setSelectedCamData([]);
    setShowClipSwap(false);
    setShowMapSwap(false);
    setShowProVideoSwap(true);
  };

  const handleMapClipSelect = (camData) => {
    setSelectedPersonData([]);
    setSelectedVideoUrl('');
    setSelectedCamData(camData);
    setShowClipSwap(false);
    setShowMapSwap(true);
    setShowProVideoSwap(false);
  };

  const handleClipSwapClick = (videoUrl) => {
    setSelectedVideoUrl(videoUrl);
    setShowClipSwap(true);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
  };

  const resetSelection = () => {
    setSelectedPersonData([]);
    setSelectedVideoUrl('');
    setSelectedCamData([]);
    setShowClipSwap(false);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
  };

  return (
    <div>
      <Navbar resetSelection={resetSelection} />
      <div className="Main">
        <SidebarLeft onAddressSelect={handleMapClipSelect} />

        {/* 조건에 따라 컴포넌트 렌더링 */}
        {showMapSwap && <MapSwap selectedCamData={selectedCamData} />}
        {showProVideoSwap && <ProVideoSwap personData={selectedPersonData} videoUrl={selectedVideoUrl} />}
        {showClipSwap && <ClipSwap videoUrl={selectedVideoUrl} />}
        {!showMapSwap && !showProVideoSwap && !showClipSwap && <Content />}
        
        <SidebarRight onPersonSelect={handlePersonSelect} />
      </div>
      <Footer onClipSwapClick={handleClipSwapClick} />
    </div>
  );
}

export default Home;
