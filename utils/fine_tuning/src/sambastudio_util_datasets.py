import os
import json
import yaml
from snsdk import SnSdk
import logging

SNAPI_PATH = "~/.snapi"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


class SnsdkWrapper:
    """init"""

    def __init__(self, config_path=None) -> None:

        self.config = self._load_config(config_path)
        self.snapi_path = self.config["sambastudio"]["snapi_path"]

        if self.snapi_path is None or len(self.snapi_path) == 0:
            self.snapi_path = SNAPI_PATH

        host_url, tenant_id, access_key = self._get_sambastudio_variables()

        self.snsdk_client = SnSdk(
            host_url=host_url,
            access_key=access_key,
            tenant_id=tenant_id,
        )

        tenant = self.search_tenant(tenant_id)
        if tenant is not None:
            self.tenant_name = tenant["tenant_name"]
        else:
            raise ValueError(f"Tenant {tenant_id} not found")

    """Tenant methods"""

    def list_tenants(self) -> list | None:
        """Lists all tenants

        Returns:
            list | None: list of existing tenants. If there's an error, None is returned.
        """
        response = self.snsdk_client.list_tenants()
        if response["status_code"] == 200:
            return response["data"]
        else:
            logging.error(
                f"Unexpected error happened in Snsdk client. Returning None. Status code: {response['status_code']}"
            )
            return None

    def search_tenant(self, tenant_id: str) -> str | None:
        """Search tenant in list of tenants

        Args:
            tenant_id (str): tenant id to search

        Returns:
            str: searched tenant information. If there's an error, None is returned.
        """
        tenants = self.list_tenants()
        for tenant in tenants:
            if tenant["tenant_id"] == tenant_id:
                return tenant
        logging.info(f"Tenant {tenant_id} not found. Returning None.")
        return None

    """Internal methods"""

    def _get_sambastudio_variables(self) -> tuple:
        """Gets Sambastudio host name, tenant id and access key from Snapi folder location

        Raises:
            FileNotFoundError: raises error when the snapi config or secret file is not found
            ValueError: raises error when the snapi config file doesn't contain a correct json format

        Returns:
            tuple: host name, tenant id and access key from snapi setup
        """
        snapi_config = ""
        snapi_secret = ""

        try:

            # reads snapi config json
            snapi_config_path = os.path.expanduser(self.snapi_path) + "/config.json"
            with open(snapi_config_path, "r") as file:
                snapi_config = json.load(file)

            # reads snapi secret txt file
            snapi_secret_path = os.path.expanduser(self.snapi_path) + "/secret.txt"
            with open(snapi_secret_path, "r") as file:
                snapi_secret = file.read()

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: The file {snapi_config_path} does not exist."
            )
        except ValueError:
            raise ValueError(
                f"Error: The file {snapi_config_path} contains invalid JSON."
            )

        host_name = snapi_config["HOST_NAME"]
        tenant_id = snapi_config["TENANT_ID"]
        access_key = snapi_secret

        return host_name, tenant_id, access_key

    def _load_config(self, file_path):
        """Loads a YAML configuration file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: The configuration data loaded from the YAML file.
        """
        try:
            with open(file_path, "r") as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file {file_path} does not exist.")
        except yaml.scanner.ScannerError:
            raise ValueError(f"Error: The file {file_path} contains invalid yaml.")
        return config

    """Dataset"""

    def search_dataset(self, dataset_name):
        dataset_keys = ["id", "dataset_name", "metadata"]

        dataset_results = self.snsdk_client.list_datasets()

        for ds in dataset_results["datasets"]:
            if ds["dataset_name"].upper() == dataset_name.upper():
                dataset = {key: ds[key] for key in dataset_keys}
        return dataset["id"]

    def delete_dataset(self, dataset_name):
        pass

    def create_dataset(self, path):
        # search first, if it already exists, return id, if not create it
        # to be generic, in theory it should be to upload files from a local folder
        # if creation outputs error, show it and stop e2e process
        pass

    def check_dataset_creation_progress(self, dataset_name):
        # check progress in case data is huge
        # e2e process has to wait while this is working, showing progress
        pass

    """generic stuff"""

    def list_models():
        pass

    def list_projects():
        pass

    def list_datasets():
        pass


if __name__ == "__main__":

    config_path = "../config.yaml"
    snsdkwrapper_client = SnsdkWrapper(config_path=config_path)

    print(snsdkwrapper_client.list_tenants())
