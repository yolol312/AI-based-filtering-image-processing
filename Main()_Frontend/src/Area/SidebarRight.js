import React, { useContext, useState, useEffect } from 'react';
import {
  Box,
  Card,
  Table,
  TableBody,
  TableRow,
  TableCell,
  Typography,
  Avatar,
  Button,
} from '@mui/material';
import { PersonDataContext } from '../Data/PersonDataContext';
import { DataContext } from '../Data/DataContext';
import { MapDataContext } from '../Data/MapDataContext';
import { ClipDataContext } from '../Data/ClipDataContext';
import './SidebarRight.css';

const Server_IP = process.env.REACT_APP_Server_IP;

function SidebarRight({ onAddressSelect, onPersonClickInSidebarRight }) {
  const { data } = useContext(DataContext);
  const { personData = [], setPersonData } = useContext(PersonDataContext);
  const { clipdata = [], setClipInfo } = useContext(ClipDataContext);
  const { updateMapInfo, triggerRouteAnimation, MapInfo, resetRouteAnimation, isInitialMapInfo } =
    useContext(MapDataContext);
  const userId = data.userId;
  const [address, setAddress] = useState('');
  const [trigger, setTrigger] = useState(false);

  useEffect(() => {
    if (address) {
      console.log('Address updated:', address);
      handleAddressClick(address);
    }
  }, [trigger]);

  useEffect(() => {
    if (isInitialMapInfo()) {
      if (triggerRouteAnimation) {
        resetRouteAnimation(); // 트리거 상태를 초기화하는 작업
      }
    }
  }, [MapInfo]);

  const handleAddressClick = async (address) => {
    const selectedCameras = data.mapcameraprovideoinfo.filter((cam) => cam.address === address);
    onAddressSelect(selectedCameras);
  };

  const groupedData = personData.reduce((acc, person) => {
    const videoName = person.pro_video_name;
    if (!acc[videoName]) {
      acc[videoName] = [];
    }
    acc[videoName].push(person);
    return acc;
  }, {});

  const handlePersonClick = async (person) => {
    try {
      const response = await fetch(
        `${Server_IP}/get_Person_to_clip?user_id=${userId}&person_id=${person.person_id}&filter_id=${person.filter_id}`,
      );
      if (response.ok) {
        const result = await response.json();
        setClipInfo(result.clip_info);
        onPersonClickInSidebarRight(result.clip_info); // 전달받은 핸들러 호출
        console.log('clip data saved:', result);
      } else {
        console.error('Failed to fetch person data');
      }
    } catch (error) {
      console.error('Error fetching person data:', error);
    }
  };

  const handlePathClick = async (personId) => {
    try {
      const response = await fetch(`${Server_IP}/map_cal?person_id=${personId}`);
      const responseData = await response.json();
      if (response.ok) {
        setAddress(responseData.address[0].address); // 상태 업데이트
        setTrigger((prev) => !prev); // 트리거 상태를 변경하여 useEffect를 실행
        updateMapInfo(responseData);
        console.log('Map data saved:', responseData);
      } else {
        console.error('Failed to fetch map data:', responseData);
      }
    } catch (error) {
      console.error('Error fetching map data:', error);
    }
  };

  const mapInfoAddresses = address;
  console.log('Address saved1:', mapInfoAddresses);

  return (
    <Box className="SidebarRight">
      <Typography variant="h6" className="SidebarRighttitle">
        Person 목록
      </Typography>
      {Object.entries(groupedData).map(([videoName, persons], index) => (
        <VideoCard
          key={index}
          persons={persons}
          onPersonClick={handlePersonClick}
          
          onPathClick={handlePathClick}
          //onMapClick={handleAddressClick}
          address={mapInfoAddresses}
        />
      ))}
    </Box>
  );
}

function VideoCard({ persons, onPersonClick, onPathClick, address }) {
  const sortedPersons = persons.sort((a, b) => a.person_id.localeCompare(b.person_id));
  console.log('Address saved2:', address);

  return (
    <Card className={'card'}>
      <Table className="person-table">
        <TableBody>
          {sortedPersons.map((person, index) => (
            <TableRow key={index} className="person-row">
              <TableCell>
                <Typography>{person.person_id}</Typography>
                <Typography>나이: {person.person_age}</Typography>
                <Typography>옷: {person.person_clothes}</Typography>
                <Typography>색상: {person.person_color}</Typography>
                <Typography>성별: {person.person_gender}</Typography>
                <Avatar
                  src={`${Server_IP}/${person.person_face}`}
                  alt={person.person_id}
                  className="person-avatar"
                />
                <Button
                  className="clip-button"
                  variant="contained"
                  onClick={(e) => {
                    e.stopPropagation();
                    onPersonClick(person);
                  }}
                >
                  clip 보기
                </Button>
                <Button
                  className="path-button"
                  variant="contained"
                  onClick={(e) => {
                    e.stopPropagation();
                    onPathClick(person.person_id);
                  }}
                >
                  경로 보기
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Card>
  );
}

export default SidebarRight;
