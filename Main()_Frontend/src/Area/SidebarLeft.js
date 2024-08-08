import React, { useContext, useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import { SimpleTreeView } from '@mui/x-tree-view/SimpleTreeView';
import { TreeItem } from '@mui/x-tree-view/TreeItem';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import { DataContext } from '../Data/DataContext';
import { PersonDataContext } from '../Data/PersonDataContext';
import './SidebarLeft.css';

const SidebarLeft = ({ onAddressSelect, onPersonSelect }) => {
  const { data } = useContext(DataContext);
  const { setPersonData } = useContext(PersonDataContext);
  const [selectedAddress, setSelectedAddress] = useState(null);
  const [mapVisible, setMapVisible] = useState(false);

  const Server_IP = process.env.REACT_APP_Server_IP;
  const userId = data?.userId;

  useEffect(() => {
    console.log('Loaded data:', data);
  }, [data]);

  const handleAddressClick = (address) => {
    setSelectedAddress(address);
    const selectedCameras = data.mapcameraprovideoinfo.filter((cam) => cam.address === address);
    onAddressSelect(selectedCameras);
    setPersonData([]);
  };

  const handleTreeItemClick = async (filterId, provideoname) => {
    try {
      const response = await fetch(
        `${Server_IP}/select_person?filter_id=${filterId}&pro_video_name=${provideoname}`,
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        },
      );

      if (response.ok) {
        const result = await response.json();
        setPersonData(result.person_info);
        handleVideoNameClick(provideoname, result.person_info);
      } else {
        const error = await response.json();
        console.error('Error fetching person data:', error.error);
      }
    } catch (error) {
      console.error('Exception while fetching person data:', error);
    }
  };

  const handleVideoNameClick = async (provideoname, persons) => {
    const payload = { user_id: userId, pro_video_name: provideoname };
    const queryParams = new URLSearchParams(payload).toString();
    const videoUrl = `${Server_IP}/stream_video?${queryParams}`;

    console.log('Selected Video URL:', videoUrl);
    console.log('Persons Data:', persons);

    try {
      if (onPersonSelect && typeof onPersonSelect === 'function') {
        onPersonSelect(persons, videoUrl);
      } else {
        throw new Error('onPersonSelect is not a function');
      }
    } catch (error) {
      console.error('Error sending data to server:', error);
    }
  };

  const groupData = (data) => {
    const grouped = {};
    (data || []).forEach((video) => {
      if (!grouped[video.address]) {
        grouped[video.address] = {};
      }
      if (!grouped[video.address][video.bundle_name]) {
        grouped[video.address][video.bundle_name] = {};
      }
      const filterKey = `${video.filter_age}, ${video.filter_clothes}, ${video.filter_color}, ${video.filter_gender}`;
      if (!grouped[video.address][video.bundle_name][filterKey]) {
        grouped[video.address][video.bundle_name][filterKey] = [];
      }
      grouped[video.address][video.bundle_name][filterKey].push(video);
    });
    return grouped;
  };

  const groupedVideos = groupData(data.mapcameraprovideoinfo);

  return (
    <Box className="SidebarLeft">
      <SimpleTreeView
        aria-label="file system navigator"
        defaultCollapseIcon={<ExpandMoreIcon />}
        defaultExpandIcon={<ChevronRightIcon />}
        className="tree-view"
      >
        {Object.keys(groupedVideos).map((address, addressIndex) => (
          <TreeItem
            key={addressIndex}
            itemId={`address-${addressIndex}`}
            label={address}
            onClick={() => handleAddressClick(address)}
            className={selectedAddress === address ? 'tree-item-selected' : 'tree-item'}
          >
            {Object.keys(groupedVideos[address]).map((bundle, bundleIndex) => (
              <TreeItem
                key={`bundle-${addressIndex}-${bundleIndex}`}
                itemId={`bundle-${addressIndex}-${bundleIndex}`}
                label={bundle}
              >
                {Object.keys(groupedVideos[address][bundle]).map((filter, filterIndex) => (
                  <TreeItem
                    key={`filter-${addressIndex}-${bundleIndex}-${filterIndex}`}
                    itemId={`filter-${addressIndex}-${bundleIndex}-${filterIndex}`}
                    label={filter}
                  >
                    {groupedVideos[address][bundle][filter].map((video, videoIndex) => (
                      <TreeItem
                        key={`video-${addressIndex}-${bundleIndex}-${filterIndex}-${videoIndex}`}
                        itemId={`video-${addressIndex}-${bundleIndex}-${filterIndex}-${videoIndex}`}
                        label={video.pro_video_name}
                        onClick={() =>
                          handleTreeItemClick(video.filter_id, video.pro_video_name, video)
                        }
                      />
                    ))}
                  </TreeItem>
                ))}
              </TreeItem>
            ))}
          </TreeItem>
        ))}
      </SimpleTreeView>
    </Box>
  );
};

export default SidebarLeft;