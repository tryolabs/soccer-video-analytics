from typing import List


class Team:
    def __init__(
        self,
        name: str,
        color: tuple = (0, 0, 0),
        abbreviation: str = "NNN",
        board_color: tuple = None,
        text_color: tuple = (0, 0, 0),
    ):
        """
        Initialize Team

        Parameters
        ----------
        name : str
            Team name
        color : tuple, optional
            Team color, by default (0, 0, 0)
        abbreviation : str, optional
            Team abbreviation, by default "NNN"

        Raises
        ------
        ValueError
            If abbreviation is not 3 characters long or not uppercase
        """
        self.name = name
        self.possession = 0
        self.passes = []
        self.color = color
        self.abbreviation = abbreviation
        self.text_color = text_color

        if board_color is None:
            self.board_color = color
        else:
            self.board_color = board_color

        if len(abbreviation) != 3 or not abbreviation.isupper():
            raise ValueError("abbreviation must be length 3 and uppercase")

    def get_percentage_possession(self, duration: int) -> float:
        """
        Return team possession in percentage

        Parameters
        ----------
        duration : int
            Match duration in frames

        Returns
        -------
        float
            Team possession in percentage
        """
        if duration == 0:
            return 0
        return round(self.possession / duration, 2)

    def get_time_possession(self, fps: int) -> str:
        """
        Return team possession in time format

        Parameters
        ----------
        fps : int
            Frames per second

        Returns
        -------
        str
            Team possession in time format (mm:ss)
        """

        seconds = round(self.possession / fps)
        minutes = seconds // 60
        seconds = seconds % 60

        # express seconds in 2 digits
        seconds = str(seconds)
        if len(seconds) == 1:
            seconds = "0" + seconds

        # express minutes in 2 digits
        minutes = str(minutes)
        if len(minutes) == 1:
            minutes = "0" + minutes

        return f"{minutes}:{seconds}"

    def __str__(self):
        return self.name

    def __eq__(self, other: "Team") -> bool:
        if isinstance(self, Team) == False or isinstance(other, Team) == False:
            return False

        return self.name == other.name

    @staticmethod
    def from_name(teams: List["Team"], name: str) -> "Team":
        """
        Return team object from name

        Parameters
        ----------
        teams : List[Team]
            List of Team objects
        name : str
            Team name

        Returns
        -------
        Team
            Team object
        """
        for team in teams:
            if team.name == name:
                return team
        return None
