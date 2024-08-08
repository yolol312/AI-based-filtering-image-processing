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
import MyPageSwap from './Area/MyPageSwap';

function Home() {
  const [selectedPersonData, setSelectedPersonData] = useState([]);
  const [selectedVideoUrl, setSelectedVideoUrl] = useState('');
  const [selectedCamData, setSelectedCamData] = useState([]);
  const [showClipSwap, setShowClipSwap] = useState(false);
  const [showMapSwap, setShowMapSwap] = useState(false);
  const [showProVideoSwap, setShowProVideoSwap] = useState(false);
  const [showMyPageSwap, setShowMyPageSwap] = useState(false);

  const handlePersonSelect = (personData, videoUrl) => {
    setSelectedPersonData(personData);
    setSelectedVideoUrl(videoUrl);
    setSelectedCamData([]);
    setShowClipSwap(false);
    setShowMapSwap(false);
    setShowProVideoSwap(true);
    setShowMyPageSwap(false);
  };

  const handleMapClipSelect = (camData) => {
    setSelectedPersonData([]);
    setSelectedVideoUrl('');
    setSelectedCamData(camData);
    setShowClipSwap(false);
    setShowMapSwap(true);
    setShowProVideoSwap(false);
    setShowMyPageSwap(false);
  };

  const handleClipSwapClick = (videoUrl) => {
    setSelectedVideoUrl(videoUrl);
    setShowClipSwap(true);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
    setShowMyPageSwap(false);
  };

  const handlePersonClickInSidebarRight = (clipInfo) => {
    setSelectedPersonData(clipInfo);
    setShowClipSwap(false);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
    setShowMyPageSwap(false);
  };

  const handleMyPageClick = () => {
    setSelectedPersonData([]);
    setSelectedVideoUrl('');
    setSelectedCamData([]);
    setShowClipSwap(false);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
    setShowMyPageSwap(true);
  };

  const resetSelection = () => {
    setSelectedPersonData([]);
    setSelectedVideoUrl('');
    setSelectedCamData([]);
    setShowClipSwap(false);
    setShowMapSwap(false);
    setShowProVideoSwap(false);
    setShowMyPageSwap(false);
  };

  return (
    <div>
      <Navbar resetSelection={resetSelection} onMyPageClick={handleMyPageClick} />
      <div className="Main">
        <SidebarLeft onAddressSelect={handleMapClipSelect} onPersonSelect={handlePersonSelect} />
        {showMapSwap && <MapSwap selectedCamData={selectedCamData} />}
        {showProVideoSwap && (
          <ProVideoSwap personData={selectedPersonData} videoUrl={selectedVideoUrl} />
        )}
        {showClipSwap && <ClipSwap videoUrl={selectedVideoUrl} />}
        {showMyPageSwap && <MyPageSwap />}
        {!showMapSwap && !showProVideoSwap && !showClipSwap && !showMyPageSwap && <Content />}
        <SidebarRight
          onAddressSelect={handleMapClipSelect}
          onPersonClickInSidebarRight={handlePersonClickInSidebarRight}
        />
      </div>
      <Footer onClipSwapClick={handleClipSwapClick} clipInfo={selectedPersonData} />
    </div>
  );
}

export default Home;
